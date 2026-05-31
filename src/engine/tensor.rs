use std::{array, fmt::Display};

use crate::engine::value::Value;

#[derive(Clone)]
pub struct Tensor<const ROW: usize, const COL: usize> {
    data: [[Value; COL]; ROW],
}

impl<const ROW: usize, const COL: usize> Tensor<ROW, COL> {
    pub fn new(data: [[Value; COL]; ROW]) -> Self {
        Self { data }
    }

    pub fn matmul<const RCOL: usize>(&self, rhs: Tensor<COL, RCOL>) -> Tensor<ROW, RCOL> {
        let mut result = array::from_fn(|_| array::from_fn(|_| Value::new(0.0)));

        for i in 0..ROW {
            for j in 0..RCOL {
                let mut sum = Value::new(0.0);
                for k in 0..COL {
                    sum = sum + (self.data[i][k].clone() * rhs.data[k][j].clone());
                }
                result[i][j] = sum;
            }
        }

        Tensor { data: result }
    }

    pub fn shape(&self) -> (usize, usize) {
        (ROW, COL)
    }

    pub fn softmax(&self) -> Tensor<ROW, COL> {
        let mut result = array::from_fn(|_| array::from_fn(|_| Value::new(0.0)));

        let mut row_max_vec: Vec<Value> = Vec::new();

        for i in 0..ROW {
            let mut row_max = self.data[i][0].clone();
            let mut row_max_inner = row_max.inner.lock().unwrap().data;

            for j in 0..COL {
                let inner_data = self.data[i][j].inner.lock().unwrap().data;
                if row_max_inner < inner_data {
                    row_max_inner = inner_data;
                    row_max = self.data[i][j].clone();
                }
            }

            row_max_vec.push(row_max);
        }

        let mut max_adjusted_tensor: [[Value; COL]; ROW] =
            array::from_fn(|_| array::from_fn(|_| Value::new(0.0)));
        let mut row_sum_vec: Vec<Value> = Vec::new();

        for i in 0..ROW {
            let mut row_sum = Value::new(0.0);
            for j in 0..COL {
                let max_adjusted_data = self.data[i][j].clone() - row_max_vec[i].clone();
                let exp_node = max_adjusted_data.exp();
                max_adjusted_tensor[i][j] = exp_node.clone();
                row_sum = row_sum + exp_node;
            }
            row_sum_vec.push(row_sum);
        }

        for i in 0..ROW {
            for j in 0..COL {
                result[i][j] = max_adjusted_tensor[i][j].clone() / row_sum_vec[i].clone();
            }
        }

        Tensor { data: result }
    }

    pub fn cross_entropy(&self, target: Tensor<ROW, COL>) -> Tensor<1, 1> {
        let mut total_loss = Value::new(0.0);

        for i in 0..ROW {
            for j in 0..COL {
                let y_true = target.data[i][j].clone();
                let y_pred = self.data[i][j].clone();
                let epsilon = Value::new(1e-7);

                let safe_pred = y_pred + epsilon;
                let term = y_true * safe_pred.ln();
                total_loss = total_loss + term
            }
        }

        total_loss = Value::new(-1.0) * total_loss;
        total_loss = total_loss / Value::new(ROW as f32);

        Tensor::new([[total_loss]])
    }

    pub fn backward(&self) {
        assert!(
            ROW == 1 && COL == 1,
            "Back Propagation on non-scaler Tensor is not allowed."
        );

        self.data[0][0].back_prop();
    }

    pub fn zero_grad(&self) {
        for i in 0..ROW {
            for j in 0..COL {
                self.data[i][j].zero_grad();
            }
        }
    }

    pub fn step(&self, learning_rate: f32) {
        for i in 0..ROW {
            for j in 0..COL {
                let mut inner = self.data[i][j].inner.lock().unwrap();

                inner.data -= learning_rate * inner.grad;
            }
        }
    }

    pub fn data(&self) -> f32 {
        assert!(ROW == 1 && COL == 1, "Dim must be 1x1");

        self.data[0][0].inner.lock().unwrap().data
    }

    pub fn graph(&self) -> String {
        assert!(ROW == 1 && COL == 1, "Cannot graph n-d tensors");

        self.data[0][0]._graph()
    }
}

impl<const ROW: usize, const COL: usize> Display for Tensor<ROW, COL> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut display_text = String::new();

        for i in &self.data {
            for j in i {
                display_text += format!("({}) ", j).as_str();
            }
            display_text += "\n";
        }

        write!(f, "{}", display_text)
    }
}

impl<const ROW: usize, const COL: usize> TryFrom<Vec<Tensor<1, COL>>> for Tensor<ROW, COL> {
    type Error = &'static str;

    fn try_from(vec_2d: Vec<Tensor<1, COL>>) -> Result<Self, Self::Error> {
        if vec_2d.len() != ROW {
            panic!("Vector row count does not match with Tensor row count.");
        }

        let mut row_iter = vec_2d.into_iter();

        let data: [[Value; COL]; ROW] = array::from_fn(|_| {
            let single_tensor = row_iter.next().unwrap();
            let [actual_row] = single_tensor.data;
            actual_row
        });

        Ok(Tensor { data })
    }
}
