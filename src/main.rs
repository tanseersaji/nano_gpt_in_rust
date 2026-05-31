mod engine;
mod loaders;
mod models;

use loaders::data::DataLoader;
use models::bigram::Bigram;

fn main() {
    const BATCH_SIZE: usize = 1000;

    let loader = DataLoader::load_data("./dataset.txt").unwrap();
    let learning_rate = 0.01;
    let epoch = 1000;
    let steps = loader.total_steps::<BATCH_SIZE>();
    let model = Bigram::<BATCH_SIZE, 30>::new();

    println!("Total Steps = {steps}");

    for i in 0..epoch {
        let mut step_loss: f32 = 0.0;
        for step in 0..steps {
            let (train_x, train_y) = loader.load_batches::<BATCH_SIZE>(step);

            let out = model.forward(train_x);
            let loss = out.cross_entropy(train_y);

            model.weights.zero_grad();
            loss.backward();
            model.weights.step(learning_rate);

            step_loss += loss.data();
            println!("Loss = {loss}");

            // if step % 100 == 0 {
            //     println!("Loss = {loss}");
            // }
        }

        step_loss /= steps as f32;
        if i % 100 == 0 {
            println!("Step Loss = {step_loss}");
        }
    }
}
