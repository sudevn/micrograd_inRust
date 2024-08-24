use micrograd::{generate_chart, propagate, Value};
use rand::Rng;
use std::fmt;

#[derive(Debug)]
struct Neuron {
    weight: Vec<Value>,
    bias: Value,
}

impl Neuron {
    fn new(nin: i64, neuron_name: &str) -> Neuron {
        let mut rng = rand::thread_rng();
        let weight = (0..nin)
            .enumerate()
            .map(|(i, _)| {
                Value::new(
                    rng.gen_range(-1.00..1.00),
                    format!("{}_weight_{:.2}", neuron_name, i),
                )
            })
            .collect();
        let bias = Value::new(rng.gen_range(-1.00..1.00), format!("{}_bias", neuron_name));
        Self { weight, bias }
    }
    fn n(self, x: &[f64]) -> Value {
        // Calculate weighted sum
        let mut weighted_sum: Value = Value::new(0.00, "init".to_string());
        for (w, &x_val) in self.weight.into_iter().zip(x.into_iter()) {
            weighted_sum = weighted_sum + (w * Value::new(x_val, "neurons".to_string()));
        }

        // Add bias
        let _result = weighted_sum + Value::new(self.bias.data, "Bias".to_string());
        let result = _result.tanh();

        // Return the result
        result
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Weight: [")?;
        for (i, value) in self.weight.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", value.data)?;
        }
        write!(f, "], Bias: {})", self.bias.data)
    }
}

fn main() {
    // Init value
    // let x1 = Value::new(4.00, "x1".to_owned());
    // let x2 = Value::new(0.00, "x2".to_string());
    // //init weights
    // let w1 = Value::new(-3.00, "w1".to_string());
    // let w2 = Value::new(5.00, "w2".to_string());
    // //init bias for neurons
    // let b = Value::new(6.88137356872, "Bias".to_string());

    // let n = x1 * w1 + x2 * w2 + b;

    // let mut o = n.tanh();

    // propagate(&mut o);
    // println!("{}", generate_chart(&o, 1));
    let x = [2.00, 3.00];
    let n = Neuron::new(2, "test");
    let mut y = n.n(&x);
    propagate(&mut y);
    println!("{}", generate_chart(&y, 0));
}
