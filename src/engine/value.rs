use std::{
    collections::HashSet,
    fmt::Display,
    iter::Sum,
    ops::{Add, Div, Mul, Sub},
    sync::{Arc, Mutex},
};

pub struct ValueData {
    pub data: f32,
    pub grad: f32,
    parents: Vec<Arc<Mutex<ValueData>>>,
    op: String,
    _backward: Option<Arc<dyn Fn() + Send + Sync>>,
}

#[derive(Clone)]
pub struct Value {
    pub inner: Arc<Mutex<ValueData>>,
}

impl Value {
    pub fn new(data: f32) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ValueData {
                data,
                grad: 0.0,
                parents: Vec::new(),
                op: String::new(),
                _backward: None,
            })),
        }
    }

    fn topalogical_sort(&self) -> Vec<Value> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();

        self.build_topo(&mut result, &mut visited);

        result
    }

    fn build_topo(&self, result: &mut Vec<Value>, visited: &mut std::collections::HashSet<usize>) {
        let ptr = Arc::as_ptr(&self.inner) as usize;

        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        let parents = self.inner.lock().unwrap().parents.clone();

        result.push(Value {
            inner: Arc::clone(&self.inner),
        });

        for parent in parents {
            let parent_value = Value {
                inner: Arc::clone(&parent),
            };

            parent_value.build_topo(result, visited);
        }
    }

    pub fn back_prop(&self) {
        self.inner.lock().unwrap().grad = 1.0;

        let topo = self.topalogical_sort();

        for val in topo {
            let f = val.inner.lock().unwrap()._backward.clone();
            if let Some(f) = f {
                f();
            }
        }
    }

    pub fn _graph(&self) -> String {
        let mut dot = String::from("digraph {\n    rankdir=LR;\n");
        let mut visited = std::collections::HashSet::new();
        self._build_graph(&mut dot, &mut visited);
        dot.push_str("}");
        dot
    }

    fn _build_graph(&self, dot: &mut String, visited: &mut std::collections::HashSet<usize>) {
        let ptr = Arc::as_ptr(&self.inner) as usize;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        let inner = self.inner.lock().unwrap();

        // Node label
        dot.push_str(&format!(
            "    n{} [label=\"data={:.2} | grad={:.2}\" shape=record];\n",
            ptr, inner.data, inner.grad
        ));

        // Op node + edges
        if !inner.op.is_empty() {
            let op_id = format!("op{}", ptr);
            dot.push_str(&format!("    {} [label=\"{}\"];\n", op_id, inner.op));
            dot.push_str(&format!("    {} -> n{};\n", op_id, ptr));

            // Recurse into parents
            let parents = inner.parents.clone();
            drop(inner); // release lock before recursing

            for parent in &parents {
                let parent_ptr = Arc::as_ptr(&parent) as usize;
                dot.push_str(&format!("    n{} -> {};\n", parent_ptr, op_id));
                let parent_value = Value {
                    inner: Arc::clone(parent),
                };
                parent_value._build_graph(dot, visited);
            }
        }
    }

    pub fn tanh(&self) -> Value {
        let out = self.inner.lock().unwrap().data.tanh();
        let out_value = Value::new(out);

        let out_inner = Arc::clone(&out_value.inner);
        let self_inner = Arc::clone(&self.inner);

        let backward = move || {
            let out_grad = out_inner.lock().unwrap().grad;
            let tanh_deriv = 1.0 - out_inner.lock().unwrap().data.powi(2);
            self_inner.lock().unwrap().grad += out_grad * tanh_deriv;
        };

        {
            let mut inner = out_value.inner.lock().unwrap();
            inner.op = "tanh".to_string();
            inner.parents = vec![Arc::clone(&self.inner)];
            inner._backward = Some(Arc::new(backward));
        }

        out_value
    }

    pub fn powi(self, exp: i32) -> Value {
        let out = self.inner.lock().unwrap().data.powi(exp);
        let out_value = Value::new(out);

        let out_inner = Arc::clone(&out_value.inner);
        let self_inner = Arc::clone(&self.inner);

        let backward = move || {
            let pow_deriv = (exp as f32) * self_inner.lock().unwrap().data.powi(exp - 1);
            self_inner.lock().unwrap().grad = pow_deriv * out_inner.lock().unwrap().grad;
        };

        {
            let mut inner = out_value.inner.lock().unwrap();
            inner.op = format!("powi({})", exp).to_string();
            inner.parents = vec![Arc::clone(&self.inner)];
            inner._backward = Some(Arc::new(backward));
        }

        out_value
    }

    pub fn mse(a: Vec<Value>, b: Vec<Value>) -> Result<Value, String> {
        if a.len() != b.len() {
            return Err(format!(
                "Length of a and b must be same found {} and {}",
                a.len(),
                b.len()
            ));
        }

        let sq_error: Value = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| (b.clone() - a.clone()).powi(2))
            .sum();

        let mean_sq_error = sq_error / Value::new(a.len() as f32);

        Ok(mean_sq_error)
    }

    pub fn zero_grad(&self) {
        self.inner.lock().unwrap().grad = 0.0;
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        let data = self.inner.lock().unwrap().data + rhs.inner.lock().unwrap().data;
        let new_value = Value::new(data);

        let self_inner = Arc::clone(&self.inner);
        let rhs_inner = Arc::clone(&rhs.inner);
        let new_value_inner = Arc::clone(&new_value.inner);

        let backward = move || {
            let new_value_grad = new_value_inner.lock().unwrap().grad;
            self_inner.lock().unwrap().grad += new_value_grad;
            rhs_inner.lock().unwrap().grad += new_value_grad;
        };

        {
            let mut inner = new_value.inner.lock().unwrap();
            inner.op = "+".to_string();
            inner.parents = vec![Arc::clone(&self.inner), Arc::clone(&rhs.inner)];
            inner._backward = Some(Arc::new(backward));
        }

        Value {
            inner: new_value.inner,
        }
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        let data = self.inner.lock().unwrap().data - rhs.inner.lock().unwrap().data;
        let new_value = Value::new(data);

        let self_inner = Arc::clone(&self.inner);
        let rhs_inner = Arc::clone(&rhs.inner);
        let new_value_inner = Arc::clone(&new_value.inner);

        let backward = move || {
            let new_value_grad = new_value_inner.lock().unwrap().grad;
            self_inner.lock().unwrap().grad += new_value_grad;
            rhs_inner.lock().unwrap().grad -= new_value_grad;
        };

        {
            let mut inner = new_value.inner.lock().unwrap();
            inner.op = "-".to_string();
            inner.parents = vec![Arc::clone(&self.inner), Arc::clone(&rhs.inner)];
            inner._backward = Some(Arc::new(backward));
        }

        Value {
            inner: new_value.inner,
        }
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.inner.lock().unwrap().data * rhs.inner.lock().unwrap().data;
        let new_value = Value::new(data);

        let self_inner = Arc::clone(&self.inner);
        let rhs_inner = Arc::clone(&rhs.inner);
        let new_value_inner = Arc::clone(&new_value.inner);

        let backward = move || {
            let new_grad = new_value_inner.lock().unwrap().grad;
            let self_data = self_inner.lock().unwrap().data;
            let rhs_data = rhs_inner.lock().unwrap().data;

            self_inner.lock().unwrap().grad += new_grad * rhs_data;
            rhs_inner.lock().unwrap().grad += new_grad * self_data;
        };

        {
            let mut inner = new_value.inner.lock().unwrap();

            inner.op = "*".to_string();
            inner.parents = vec![Arc::clone(&self.inner), Arc::clone(&rhs.inner)];
            inner._backward = Some(Arc::new(backward));
        }

        Value {
            inner: new_value.inner,
        }
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        let data = self.inner.lock().unwrap().data / rhs.inner.lock().unwrap().data;
        let new_value = Value::new(data);

        let self_inner = Arc::clone(&self.inner);
        let rhs_inner = Arc::clone(&rhs.inner);
        let new_value_inner = Arc::clone(&new_value.inner);

        let backward = move || {
            let new_grad = new_value_inner.lock().unwrap().grad;
            let self_data = self_inner.lock().unwrap().data;
            let rhs_data = rhs_inner.lock().unwrap().data;

            self_inner.lock().unwrap().grad += new_grad * (1.0 / rhs_data);
            rhs_inner.lock().unwrap().grad += new_grad * (-self_data / rhs_data.powi(2));
        };

        {
            let mut inner = new_value.inner.lock().unwrap();

            inner.op = "/".to_string();
            inner.parents = vec![Arc::clone(&self.inner), Arc::clone(&rhs.inner)];
            inner._backward = Some(Arc::new(backward));
        }

        Value {
            inner: new_value.inner,
        }
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Value::new(0.0), |acc, x| acc + x)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock().unwrap();
        write!(f, "Data: {} | Grad: {}", inner.data, inner.grad)
    }
}
