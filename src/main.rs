use std::borrow::BorrowMut;
use std::fmt;
use std::ops::{Add, Mul};

#[derive(Debug)]
enum TValue<'a> {
    Flt(f64),
    Norm(Value<'a>),
    RNorm(&'a Value<'a>),
}

#[derive(Debug)]
struct Value<'a> {
    data: f64,
    grad: f64,
    // _backward: Box<dyn FnMut()>,
    _prev: Option<Vec<TValue<'a>>>,
    _op: Option<String>,
}

impl<'a> Value<'a> {
    fn new(data: f64) -> Self {
        let value = Value {
            data,
            grad: 0.0000,
            _prev: None,
            _op: None,
        };
        value
    }
}
//Formating
impl<'a> fmt::Display for TValue<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TValue::Norm(data) => write!(f, "({}", data),
            TValue::RNorm(data) => write!(f, "({}", data),
            TValue::Flt(data) => write!(f, "(Value: {})", data),
        }
    }
}
impl<'a> fmt::Display for Value<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Value: {:.4}, grad: {}, Prev: ", self.data, self.grad)?;
        if let Some(prev) = &self._prev {
            for (index, item) in prev.iter().enumerate() {
                write!(f, "{},", item)?;
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
impl<'a, T> Add<T> for Value<'a>
where
    T: Into<Option<f64>>,
{
    type Output = Self;
    fn add(self, other: T) -> Self::Output {
        let other_data = other.into().unwrap_or(0.0);
        Self {
            data: self.data + other_data,
            grad: 0.0000,
            _prev: Some(vec![TValue::Norm(self), TValue::Flt(other_data)]),
            _op: Some("+".to_string()),
        }
    }
}
impl<'a> Add<&'a Value<'a>> for Value<'a> {
    type Output = Self;
    fn add(self, other: &'a Value<'a>) -> Self::Output {
        Self {
            data: self.data + other.data,
            grad: 0.0000,
            _prev: Some(vec![TValue::Norm(self), TValue::RNorm(other)]),
            _op: Some("+".to_string()),
        }
    }
}
impl<'a> Add for Value<'a> {
    type Output = Value<'a>;
    fn add(self, other: Self) -> Self::Output {
        Value {
            data: self.data + other.data,
            grad: 0.0000,
            _prev: Some(vec![TValue::Norm(self), TValue::Norm(other)]),
            _op: Some("+".to_string()),
        }
    }
}

impl<'a> Add for &'a Value<'a> {
    type Output = Value<'a>;
    fn add(self, other: Self) -> Self::Output {
        Value {
            data: self.data + other.data,
            grad: 0.0000,
            _prev: Some(vec![TValue::RNorm(self), TValue::RNorm(other)]),
            _op: Some("+".to_string()),
        }
    }
}
impl<'a> Add<Value<'a>> for &'a Value<'a> {
    type Output = Value<'a>;
    fn add(self, other: Value<'a>) -> Self::Output {
        Value {
            data: self.data + other.data,
            grad: 0.0000,
            _prev: Some(vec![TValue::RNorm(self), TValue::Norm(other)]),
            _op: Some("+".to_string()),
        }
    }
}
impl<'a, T> Add<T> for &'a Value<'a>
where
    T: Into<f64>,
{
    type Output = Value<'a>;
    fn add(self, other: T) -> Self::Output {
        let o = other.into();
        Value {
            data: self.data + &o,
            grad: 0.0000,
            _prev: Some(vec![TValue::RNorm(self), TValue::Flt(o)]),
            _op: Some("+".to_string()),
        }
    }
}

//Implementing multiply
// impl<T> Mul<T> for Value
// where
//     T: Into<f64>,
// {
//     type Output = Self;
//     fn mul(self, other: T) -> Self::Output {
//         Self {
//             data: self.data * other.into(),
//             _prev: Some(vec![TValue::Norm(self), TValue::Flt(other.into())]),
//         }
//     }
// }
// impl Mul<&Value> for Value {
//     type Output = Self;
//     fn mul(self, other: &Value) -> Self::Output {
//         Self {
//             data: self.data * other.data,
//             _prev: vec![self.data, other.data],
//         }
//     }
// }
// impl Mul for Value {
//     type Output = Value;
//     fn mul(self, other: Self) -> Self::Output {
//         Value {
//             data: self.data * other.data,
//             _prev: vec![self.data, other.data],
//         }
//     }
// }

// impl Mul for &Value {
//     type Output = Value;
//     fn mul(self, other: Self) -> Self::Output {
//         Value {
//             data: self.data * other.data,
//             _prev: vec![self.data, other.data],
//         }
//     }
// }
// impl Mul<Value> for &Value {
//     type Output = Value;
//     fn mul(self, other: Value) -> Self::Output {
//         Value {
//             data: self.data * other.data,
//             _prev: vec![self.data, other.data],
//         }
//     }
// }
// impl<T> Mul<T> for &Value
// where
//     T: Into<f64>,
// {
//     type Output = Value;
//     fn mul(self, other: T) -> Self::Output {
//         Value {
//             data: self.data * other.into(),
//             _prev: vec![self.data, other.into()],
//         }
//     }
// }
//     T: Into<f64>,
// {
//     type Output = Value;
//     fn add(self, other: T) -> Self::Output {
//         Value {
//             data: self.data + other.into(),
//             _prev: Some(vec![TValue::RNorm(self), TValue::Flt(other.into())]),
//         }
//     }
// }

fn indent(level: usize) -> String {
    const INDENT: &str = "    ";
    INDENT.repeat(level)
}

fn generate_chart<'a>(value: &'a Value<'a>, level: usize) -> String {
    let mut result = String::new();
    if let Some(ope) = &value._op {
        result.push_str(&format!(
            "{}├─{}(Value: {}, grad: {})\n",
            indent(level),
            ope,
            value.data,
            value.grad
        ));
    } else {
        result.push_str(&format!(
            "{}├─=(Value: {}, grad: {})\n",
            indent(level),
            value.data,
            value.grad
        ));
    }
    if let Some(prev) = &value._prev {
        for item in prev {
            match item {
                TValue::Norm(sub_value) => {
                    result.push_str(&generate_chart(sub_value, level + 1));
                }
                TValue::RNorm(rsub_value) => {
                    result.push_str(&generate_chart(rsub_value, level + 1));
                }
                TValue::Flt(data) => {
                    result.push_str(&format!("{}├─+(Value: {})\n", indent(level + 1), data));
                }
            }
        }
    }
    result
}

//Backpropagation
fn propagate(value: &mut Value) {
    value.grad = 1.0000;
    if let Some(prev) = &value._prev {
        for i in prev.iter() {
            print!("{}", i)
        }
    }
    print!("No previous");
}

fn f(x: f64) -> f64 {
    3.00 * f64::powf(x, 2.00) - 4.00 * x + 5.00
}

fn main() {
    let mut a = Value::new(2.00);
    let mut b = Value::new(-3.00);
    let mut c = Value::new(10.00);
    let mut d = &a + c + &b + 1.00;
    // let result = f(x + h);
    // let num = f(x);
    // let diff = (result - num) / h;
    // println!("a: {} b: {} d(a+b): {}", a, b, d)
    //propagate(&mut d);
    println!("{}", generate_chart(&d, 1));
}
