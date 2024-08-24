use micrograd::{propagate, Value};
use std::cell::RefCell;
use std::f64::consts;

#[cfg(test)]
mod tests {
    use std::borrow::Borrow;

    use super::*;

    #[test]
    fn test_forward_propagation() {
        // Test forward propagation by constructing a simple neural network
        let x1 = Value::new(4.00, "x1".to_owned());
        let x2 = Value::new(0.00, "x2".to_string());
        let w1 = Value::new(-3.00, "w1".to_string());
        let w2 = Value::new(5.00, "w2".to_string());
        let b = Value::new(6.88137356872, "Bias".to_string());

        let n = x1 * w1 + x2 * w2 + b;

        // Assert the label and data of the resulting value
        assert_eq!(n.label, "x1*w1+x2*w2+Bias");
        assert_eq!(n.data, -3.0 * 4.0 + 5.0 * 0.0 + 6.88137356872);
    }

    #[test]
    fn test_backward_propagation() {
        // Create a simple computational graph for testing backward propagation
        let x = Value::new(2.0, "x".to_string());
        let y = Value::new(3.0, "y".to_string());
        let mut z = x * y;

        // Perform backward propagation
        propagate(&mut z);
        // Access gradients from the previous nodes
        let x_grad = if let Some(x_grad_value) = z.clone()._prev {
            let _c = x_grad_value.iter().nth(0).unwrap();
            let _a: &RefCell<Value> = _c.borrow();
            _a.borrow().grad
        } else {
            panic!("No gradient found for x");
        };

        let y_grad = if let Some(y_grad_value) = z._prev {
            let _c = y_grad_value.iter().nth(1).unwrap();
            let _a: &RefCell<Value> = _c.borrow();
            _a.borrow().grad
        } else {
            panic!("No gradient found for y");
        };
        // Check gradients
        assert_eq!(x_grad, 3.0); // dz/dx = y => grad(x) = grad(z) * dz/dx = 3.0 * 1.0 = 3.0
        assert_eq!(y_grad, 2.0); // dz/dy = x => grad(y) = grad(z) * dz/dy = 3.0 * 1.0 = 3.0
    }

    #[test]
    fn test_arithmetic_operations() {
        // Test arithmetic operations
        let value1 = Value::new(2.0, "value1".to_string());
        let value2 = Value::new(3.0, "value2".to_string());

        let result_add = value1.clone() + value2.clone();
        let result_sub = value1.clone() - value2.clone();
        let result_mul = value1.clone() * value2.clone();
        let result_div = value1 / value2;

        // Assert the results of arithmetic operations
        assert_eq!(result_add.data, 5.0);
        assert_eq!(result_sub.data, -1.0);
        assert_eq!(result_mul.data, 6.0);
        assert_eq!(result_div.data, 2.0 / 3.0);
    }

    #[test]
    fn test_activation_functions() {
        // Test activation functions
        let value = Value::new(0.5, "value".to_string());
        let result_exp = value.clone().expv();
        let result_tanh = value.tanh();

        // Assert the results of activation functions
        assert_eq!(result_exp.data, consts::E.powf(0.5));
        assert_eq!(result_tanh.data, 0.46211715726000974); // Approximate value
    }
}
