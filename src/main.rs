use std::cell::RefCell;
use std::f64::consts;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Debug)]
struct Value {
    label: String,
    data: f64,
    grad: f64,
    _backward: Option<Vec<f64>>,
    _prev: Option<Vec<Rc<RefCell<Value>>>>,
    _op: Option<String>,
}

impl Value {
    fn new(data: f64, label: String) -> Self {
        Value {
            label,
            data,
            grad: 0.0000,
            _backward: None,
            _prev: None,
            _op: None,
        }
    }
    fn child(value: &Value) -> Option<&Vec<f64>> {
        if let Some(backward) = &value._backward {
            Some(backward)
        } else {
            None
        }
    }
    pub fn pow(mut self, exponent: f64) -> Self {
        let mut exp = Value::new(exponent, exponent.to_string());
        self._op = Some("^".to_string());
        exp._op = Some("^".to_string());
        Value {
            label: format!("{}^{}", self.label, exponent),
            data: self.data.powf(exponent),
            grad: 0.0,
            _backward: Some(vec![self.data, exponent]),
            _prev: Some(vec![
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(exp)),
            ]),
            _op: None,
        }
    }
    #[allow(dead_code)]
    pub fn expv(mut self) -> Self {
        let mut exp = Value::new(consts::E, "Exp".to_string());
        self._op = Some("e".to_string());
        exp._op = Some("e".to_string());
        Value {
            label: format!("Exp^{}", self.label),
            data: consts::E.powf(self.data),
            grad: 0.0,
            _backward: Some(vec![self.data, exp.data]),
            _prev: Some(vec![
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(exp)),
            ]),
            _op: None,
        }
    }
    pub fn tanh(mut self) -> Self {
        self._op = Some("Tanh".to_string());
        let e = (self.data * 2.00).exp();
        let data_o = (e + -1.00) / (e + 1.00);
        Value {
            label: format!("Tan({})", "Activation"),
            data: data_o,
            grad: 0.00,
            _backward: Some(vec![self.data]),
            _prev: Some(vec![Rc::new(RefCell::new(self))]),
            _op: None,
        }
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Value {
            label: self.label.clone(),
            data: self.data,
            grad: self.grad,
            _backward: self._backward.clone(),
            _prev: self._prev.clone(),
            _op: self._op.clone(),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Value: {:.4}, grad: {}, Prev: ", self.data, self.grad)?;
        if let Some(prev) = &self._prev {
            for (index, item) in prev.iter().enumerate() {
                let item_rc = item.borrow();
                write!(f, "{},", item_rc)?;
                if index < prev.len() - 1 {
                    write!(f, "->")?;
                }
            }
        } else {
            write!(f, "None,")?;
        }
        if let Some(ope) = &self._op {
            write!(f, " Op: {})", ope)?;
        } else {
            write!(f, " Op: None)")?;
        }

        Ok(())
    } // Displaying value with 4 decimal places
}

//Add
impl<T> Add<T> for Value
where
    T: Into<Option<f64>>,
{
    type Output = Self;
    fn add(mut self, other: T) -> Self::Output {
        let mut other_data = Value::new(other.into().unwrap_or(0.0), "const".to_string());
        self._op = Some("+".to_string());
        other_data._op = Some("+".to_string());
        Self {
            label: format!("{}+{}", self.label, other_data.label),
            data: self.data + other_data.data,
            grad: 0.0000,
            _backward: Some(vec![self.data, other_data.data]),
            _prev: Some(vec![
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(other_data)),
            ]),
            _op: None,
        }
    }
}

impl Add for Value {
    type Output = Value;
    fn add(mut self, mut other: Self) -> Self::Output {
        self._op = Some("+".to_string());
        other._op = Some("+".to_string());
        Value {
            label: format!("{}+{}", self.label, other.label),
            data: self.data + other.data,
            grad: 0.0000,
            _backward: Some(vec![self.data, other.data]),
            _prev: Some(vec![
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(other)),
            ]),
            _op: None,
        }
    }
}

//Implementing multiply
impl<T> Mul<T> for Value
where
    T: Into<Option<f64>>,
{
    type Output = Self;
    fn mul(mut self, other: T) -> Self::Output {
        let mut other_data = Value::new(other.into().unwrap_or(0.0), "const".to_string());
        self._op = Some("*".to_string());
        other_data._op = Some("*".to_string());
        Self {
            label: format!("{}*{}", self.label, other_data.label),
            data: self.data * other_data.data,
            grad: 0.0000,
            _backward: Some(vec![self.data, other_data.data]),
            _prev: Some(vec![
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(other_data)),
            ]),
            _op: None,
        }
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(mut self, mut other: Self) -> Self::Output {
        self._op = Some("*".to_string());
        other._op = Some("*".to_string());
        Value {
            label: format!("{}*{}", self.label, other.label),
            data: self.data * other.data,
            grad: 0.0000,
            _backward: Some(vec![self.data, other.data]),
            _prev: Some(vec![
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(other)),
            ]),
            _op: None,
        }
    }
}

// implement subtact
impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value {
            label: self.label,
            data: -self.data,
            grad: self.grad,
            _backward: self._backward,
            _prev: self._prev,
            _op: self._op,
        }
    }
}
impl Sub for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other) // Subtracting a value is equivalent to adding its negation
    }
}

//Implement div
impl Div for Value {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1.00)
    }
}
impl<T> Div<T> for Value
where
    T: Into<Option<f64>>,
{
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        let other = Value::new(rhs.into().unwrap_or(0.0), "const".to_string());
        self * other.pow(-1.00)
    }
}

fn indent(level: usize) -> String {
    const INDENT: &str = "    ";
    INDENT.repeat(level)
}

fn generate_chart(value: &Value, level: usize) -> String {
    let mut result = String::new();
    if let Some(ope) = &value._op {
        result.push_str(&format!(
            "{}├─{}(\x1b[93m{}\x1b[0m||Value: {}, grad: {})\n",
            indent(level),
            ope,
            value.label,
            value.data,
            value.grad
        ));
    } else {
        result.push_str(&format!(
            "{}├─=(\x1b[93m{}\x1b[0m||Value: {}, grad: {})\n",
            indent(level),
            value.label,
            value.data,
            value.grad
        ));
    }
    if let Some(prev) = &value._prev {
        for item in prev {
            let item_rc = item.borrow();
            result.push_str(&generate_chart(&item_rc, level + 1));
        }
    }
    result
}

//Backpropagation
fn propagate(value: &mut Value) {
    value.grad = 1.0000;
    backward(value);
    fn backward(bvalue: &mut Value) {
        if let Some(prev) = &bvalue._prev {
            let list = prev.iter();
            for temp in list {
                let mut i_rc = temp.borrow_mut();
                if let Some(ope) = &i_rc._op {
                    match ope.as_str() {
                        "+" => i_rc.grad = bvalue.grad,
                        "*" => {
                            if let Some(backward) = Value::child(bvalue) {
                                if let Some(other) = backward.iter().find(|x| **x != i_rc.data) {
                                    i_rc.grad = *other * bvalue.grad;
                                } else {
                                    i_rc.grad = backward.first().unwrap() * bvalue.grad;
                                }
                            }
                        }
                        "^" => {
                            if let Some(backward) = Value::child(bvalue) {
                                if let Some(other) = backward.iter().find(|x| **x != i_rc.data) {
                                    i_rc.grad =
                                        *other * (i_rc.data.powf(*other - 1.00)) * bvalue.grad;
                                } else {
                                    i_rc.grad = backward.first().unwrap()
                                        * (i_rc.data.powf(backward.first().unwrap() - 1.00))
                                        * bvalue.grad;
                                }
                            }
                        }
                        "e" => i_rc.grad = bvalue.data,
                        "Tanh" => i_rc.grad = 1.00 - bvalue.data * bvalue.data,
                        _ => println!("Function not implemented yet"),
                    }
                } else {
                    i_rc.grad = 1.0000;
                }
                backward(&mut i_rc)
            }
        }
    }
}

fn main() {
    // Init value
    let x1 = Value::new(4.00, "x1".to_owned());
    let x2 = Value::new(0.00, "x2".to_string());
    //init weights
    let w1 = Value::new(-3.00, "w1".to_string());
    let w2 = Value::new(5.00, "w2".to_string());
    //init bias for neurons
    let b = Value::new(6.88137356872, "Bias".to_string());

    let n = x1 * w1 + x2 * w2 + b;

    let mut o = n.tanh();

    propagate(&mut o);
    println!("{}", generate_chart(&o, 1));
}

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
