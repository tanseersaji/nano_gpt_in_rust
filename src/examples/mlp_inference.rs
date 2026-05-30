mod engine;

use engine::nn::MLP;
use engine::value::Value;

fn main() {
    let input_dim: usize = 2;
    let layer_dims: &[usize] = &[3, 2];
    let model = MLP::new(input_dim, layer_dims);

    let training_data: Vec<(Vec<Value>, Vec<Value>)> = vec![
        (
            vec![Value::new(0.5), Value::new(0.2)],
            vec![Value::new(0.25), Value::new(0.04)],
        ),
        (
            vec![Value::new(0.8), Value::new(0.3)],
            vec![Value::new(0.64), Value::new(0.09)],
        ),
        (
            vec![Value::new(0.1), Value::new(0.9)],
            vec![Value::new(0.01), Value::new(0.81)],
        ),
        (
            vec![Value::new(0.6), Value::new(0.4)],
            vec![Value::new(0.36), Value::new(0.16)],
        ),
        (
            vec![Value::new(0.7), Value::new(0.5)],
            vec![Value::new(0.49), Value::new(0.25)],
        ),
    ];

    let learning_rate = 0.1;
    let epoch = 1000;

    for step in 0..epoch {
        let mut total_loss = Value::new(0.0);

        for p in model.parameters() {
            p.zero_grad();
        }

        for (input_val, true_val) in &training_data {
            let out = model.forward(input_val);
            let loss = model.loss(out, true_val.to_vec());

            loss.back_prop();

            total_loss = total_loss + loss;
        }

        for p in model.parameters() {
            let grad = p.inner.lock().unwrap().grad;
            p.inner.lock().unwrap().data -= learning_rate * grad;
        }

        if step % 100 == 0 {
            println!(
                "Step - {} | Loss = {}",
                step,
                total_loss.inner.lock().unwrap().data
            );
        }
    }

    // let dot = loss.graph();
    // std::fs::write("graph.dot", &dot).unwrap();
}
