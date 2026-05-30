mod engine;
use engine::tensor::Tensor;
use engine::value::Value;

fn main() {
    // const ROW: usize = 2;
    // const COL: usize = 1;
    let b = Tensor::<2, 2>::new([
        [Value::new(6.0), Value::new(5.0)],
        [Value::new(6.0), Value::new(5.0)],
    ]);
    let a = Tensor::<2, 2>::new([
        [Value::new(-2.0), Value::new(-2.0)],
        [Value::new(0.0), Value::new(-2.0)],
    ]);

    let c = b.matmul(a);
    println!("Data:\n{}", c);

    let c = c.softmax();
    let target = Tensor::<2, 2>::new([
        [Value::new(1.0), Value::new(0.0)],
        [Value::new(0.0), Value::new(1.0)],
    ]);

    let loss = c.cross_entropy(target);
    loss.backward();

    let dot = loss.graph();
    std::fs::write("graph.dot", dot).unwrap();
}
