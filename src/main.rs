use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Mul};
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

impl<'a> Value {
    fn new(data: f64, label: String) -> Self {
        let value = Value {
            label,
            data,
            grad: 0.0000,
            _backward: None,
            _prev: None,
            _op: None,
        };
        value
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
    } // Displaying value with two decimal places
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
        other_data._op = Some("*".to_string());
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

fn indent(level: usize) -> String {
    const INDENT: &str = "    ";
    INDENT.repeat(level)
}

fn generate_chart<'a>(value: &Value, level: usize) -> String {
    let mut result = String::new();
    if let Some(ope) = &value._op {
        result.push_str(&format!(
            "{}├─{}({}||Value: {}, grad: {})\n",
            indent(level),
            ope,
            value.label,
            value.data,
            value.grad
        ));
    } else {
        result.push_str(&format!(
            "{}├─=({}||Value: {}, grad: {})\n",
            indent(level),
            value.label,
            value.data,
            value.grad
        ));
    }
    if let Some(prev) = &value._prev {
        for item in prev {
            let item_rc = item.borrow();
            result.push_str(&generate_chart(&*item_rc, level + 1));
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
            let mut i: usize = 0;
            for temp in list {
                let mut i_rc = temp.borrow_mut();
                //let mut u_rc = u.borrow_mut();
                if let Some(ope) = &i_rc._op {
                    match ope.as_str() {
                        "+" => i_rc.grad = bvalue.grad,
                        "*" => {
                            if let Some(backward) = &bvalue._backward {
                                i = i + 1;
                                if let Some(last) = backward.iter().nth(i % 2) {
                                    i_rc.grad += last * bvalue.grad;
                                }
                            }
                        }
                        _ => i_rc.grad = 0.0000,
                    }
                } else {
                    i_rc.grad = 1.0000;
                }
                backward(&mut *i_rc)
            }
        }
    }
}

fn main() {
    let a = Value::new(2.00, 'a'.to_string());
    let b = Value::new(-3.00, 'b'.to_string());
    let c = Value::new(10.00, 'c'.to_string());
    let mut d = a + c * b * 2.00 * 1.00;
    propagate(&mut d);
    println!("{}", generate_chart(&d, 1));
}
