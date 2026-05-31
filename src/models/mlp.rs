use crate::engine::core::Layer;
use crate::engine::value::Value;

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(input_dim: usize, layer_dims: &[usize]) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        let mut temp_in_dim = input_dim;

        for &layer_dim in layer_dims {
            layers.push(Layer::new(temp_in_dim, layer_dim));
            temp_in_dim = layer_dim;
        }

        Self { layers }
    }

    pub fn forward(&self, input_val: &[Value]) -> Vec<Value> {
        let mut x = input_val.to_vec();

        for layer in &self.layers {
            x = layer.forward(&x)
        }

        x
    }

    pub fn loss(&self, pred_vals: Vec<Value>, true_vals: Vec<Value>) -> Value {
        Value::mse(pred_vals, true_vals).unwrap()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
