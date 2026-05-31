use crate::engine::value::Value;
use rand::Rng;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Neuron {
    pub fn new(input_dim: usize) -> Self {
        let mut rand = rand::thread_rng();
        Self {
            weights: (0..input_dim)
                .map(|_| Value::new(rand.gen_range(-1.0..1.0)))
                .collect(),
            bias: Value::new(rand.gen_range(-1.0..1.0)),
        }
    }

    pub fn forward(&self, input_val: &[Value]) -> Value {
        let product_sum: Value = input_val
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x.clone() * w.clone())
            .sum();
        let product_sum = product_sum + self.bias.clone();
        product_sum.tanh()
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut parameters = Vec::new();
        parameters.append(&mut self.weights.clone());
        parameters.push(self.bias.clone());

        parameters
    }
}

impl Layer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            neurons: (0..output_dim).map(|_| Neuron::new(input_dim)).collect(),
        }
    }

    pub fn forward(&self, input_val: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(input_val)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}
