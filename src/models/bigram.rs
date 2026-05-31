use rand::Rng;
use std::array;

use crate::engine::{tensor::Tensor, value::Value};

pub struct Bigram<const BATCH: usize, const VOCAB_SIZE: usize> {
    pub weights: Tensor<VOCAB_SIZE, VOCAB_SIZE>,
}

impl<const BATCH: usize, const VOCAB_SIZE: usize> Bigram<BATCH, VOCAB_SIZE> {
    pub fn new() -> Self {
        let mut rand = rand::thread_rng();
        let weights: [[Value; VOCAB_SIZE]; VOCAB_SIZE] =
            array::from_fn(|_| array::from_fn(|_| Value::new(rand.gen_range(-1.0..1.0))));

        Self {
            weights: Tensor::new(weights),
        }
    }

    pub fn forward(&self, input: Tensor<BATCH, VOCAB_SIZE>) -> Tensor<BATCH, VOCAB_SIZE> {
        let out = input.matmul(self.weights.clone());
        out.softmax()
    }

    pub fn gradient_decent(&self) {}
}
