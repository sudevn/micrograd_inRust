use std::borrow::Borrow;
use std::fmt;
use std::ops::{Add, Mul};

#[derive(Debug)]
enum TValue<'a> {
    flt(f64),
    norm(Value<'a>),
    rnorm(&'a Value<'a>),
}

#[derive(Debug)]
struct Value<'a> {
    data: f64,
    // grad: f64,
    // _backward: Box<dyn FnMut()>,
    _prev: Option<Vec<TValue<'a>>>,
    // _op: String,
}

impl<'a> Value<'a> {
    fn new(data: f64) -> Self {
        let mut value = Value { data, _prev: None };
        value
    }
}
//Formating
impl<'a> fmt::Display for TValue<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TValue::norm(data) => write!(f, "({}", data),
            TValue::rnorm(data) => write!(f, "({}", data),
            TValue::flt(data) => write!(f, "(Value: {})", data),
        }
    }
}
impl<'a> fmt::Display for Value<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Value: {:.2}, Prev: ", self.data)?;
        if let Some(prev) = &self._prev {
            write!(f, "->")?;
            for (index, item) in prev.iter().enumerate() {
                write!(f, "{})", item)?;
                if index < prev.len() - 1 {
                    write!(f, "->")?;
                }
            }
        } else {
            write!(f, "None)")?;
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
            _prev: Some(vec![TValue::norm(self), TValue::flt(other_data)]),
        }
    }
}
impl<'a> Add<&'a Value<'a>> for Value<'a> {
    type Output = Self;
    fn add(self, other: &'a Value<'a>) -> Self::Output {
        Self {
            data: self.data + other.data,
            _prev: Some(vec![TValue::norm(self), TValue::rnorm(other)]),
        }
    }
}
impl<'a> Add for Value<'a> {
    type Output = Value<'a>;
    fn add(self, other: Self) -> Self::Output {
        Value {
            data: self.data + other.data,
            _prev: Some(vec![TValue::norm(self), TValue::norm(other)]),
        }
    }
}

impl<'a> Add for &'a Value<'a> {
    type Output = Value<'a>;
    fn add(self, other: Self) -> Self::Output {
        Value {
            data: self.data + other.data,
            _prev: Some(vec![TValue::rnorm(self), TValue::rnorm(other)]),
        }
    }
}
impl<'a> Add<Value<'a>> for &'a Value<'a> {
    type Output = Value<'a>;
    fn add(self, other: Value<'a>) -> Self::Output {
        Value {
            data: self.data + other.data,
            _prev: Some(vec![TValue::rnorm(self), TValue::norm(other)]),
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
            _prev: Some(vec![TValue::rnorm(self), TValue::flt(o)]),
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
//             _prev: Some(vec![TValue::norm(self), TValue::flt(other.into())]),
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
//             _prev: Some(vec![TValue::rnorm(self), TValue::flt(other.into())]),
//         }
//     }
// }

fn indent(level: usize) -> String {
    const INDENT: &str = "    ";
    INDENT.repeat(level)
}

fn generate_chart<'a>(value: &'a Value<'a>, level: usize) -> String {
    let mut result = String::new();
    result.push_str(&format!("{}├──(Value: {})\n", indent(level), value.data));
    if let Some(prev) = &value._prev {
        for item in prev {
            match item {
                TValue::norm(sub_value) => {
                    result.push_str(&generate_chart(sub_value, level + 1));
                }
                TValue::rnorm(rsub_value) => {
                    result.push_str(&generate_chart(rsub_value, level + 1));
                }
                TValue::flt(data) => {
                    result.push_str(&format!("{}├──(Value: {})\n", indent(level + 1), data));
                }
            }
        }
    }
    result
}
fn f(x: f64) -> f64 {
    3.00 * f64::powf(x, 2.00) - 4.00 * x + 5.00
}

fn main() {
    let a = Value::new(2.00);
    let b = Value::new(-3.00);
    let c = Value::new(10.00);
    let d = &a + c + &b + 1.00;
    // let result = f(x + h);
    // let num = f(x);
    // let diff = (result - num) / h;
    // println!("a: {} b: {} d(a+b): {}", a, b, d)
    println!("{}", generate_chart(&d, 1));
}
