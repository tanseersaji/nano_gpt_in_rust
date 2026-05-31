use std::{
    array,
    fs::File,
    io::{BufRead, BufReader, Error},
};

use crate::engine::{tensor::Tensor, value::Value};

const START_PAD: char = '^';
const END_PAD: char = '$';

const ITOS: [char; 30] = [
    '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', START_PAD, END_PAD,
];

// stoi can be implemented as a `const fn`, which computes at compile time!
const fn stoi(c: char) -> usize {
    match c {
        '.' => 0,
        ' ' => 27,
        START_PAD => 28,
        END_PAD => 29,
        // Using ASCII math: 'a' is 97. If we get 'a', 97 - 97 + 1 = 1.
        'a'..='z' => (c as usize) - ('a' as usize) + 1,
        _ => panic!("Character not in vocabulary!"),
    }
}

pub struct DataLoader {
    pub encoded_data: Vec<(usize, usize)>,
}

impl DataLoader {
    fn encode(line: String) -> Vec<usize> {
        let mut result: Vec<usize> = Vec::new();

        for c in line.chars() {
            result.push(stoi(c));
        }

        result
    }

    pub fn load_data(data_file_path: &str) -> Result<Self, Error> {
        let file = File::open(data_file_path)?;
        let mut reader = BufReader::new(file);
        let mut line_buf = String::new();

        let mut result: Vec<(usize, usize)> = Vec::new();

        while reader.read_line(&mut line_buf)? > 0 {
            let mut line = line_buf.clone();
            line = line
                .to_lowercase()
                .chars()
                .filter(|&c| c.is_alphabetic() || c == ' ' || c == '.')
                .collect();

            line = line.trim().to_string();

            if line.is_empty() {
                continue;
            }

            line = format!("{}{}{}", START_PAD, line, END_PAD);

            let encoded_line = Self::encode(line);

            for i in 0..encoded_line.len() - 1 {
                result.push((encoded_line[i], encoded_line[i + 1]))
            }

            line_buf.clear();
        }

        Ok(Self {
            encoded_data: result,
        })
    }

    pub fn decode(n: usize) -> char {
        ITOS[n]
    }

    fn one_hot(&self, n: usize) -> Tensor<1, 30> {
        let result: [[Value; 30]; 1] = array::from_fn(|_| {
            array::from_fn(|i| {
                if i == n {
                    Value::new(1.0)
                } else {
                    Value::new(0.0)
                }
            })
        });

        Tensor::new(result)
    }

    pub fn load_batches<const BATCH_SIZE: usize>(
        &self,
        step_count: usize,
    ) -> (Tensor<BATCH_SIZE, 30>, Tensor<BATCH_SIZE, 30>) {
        let from_idx = BATCH_SIZE * step_count;
        let to_idx = from_idx + BATCH_SIZE;

        let mut x: Vec<Tensor<1, 30>> = Vec::new();
        let mut y: Vec<Tensor<1, 30>> = Vec::new();

        for idx in from_idx..to_idx {
            let line = self.encoded_data[idx];
            x.push(self.one_hot(line.0));
            y.push(self.one_hot(line.1));
        }

        let x_tensor: Tensor<BATCH_SIZE, 30> = x.try_into().unwrap();
        let y_tensor: Tensor<BATCH_SIZE, 30> = y.try_into().unwrap();

        (x_tensor, y_tensor)
    }

    pub fn total_steps<const BATCH_SIZE: usize>(&self) -> usize {
        self.encoded_data.len() / BATCH_SIZE
    }
}
