use std::fmt;
use std::ops::{Add, Deref, Mul};

#[derive(Debug)]
struct Value {
    data: f64,
    // grad: f64,
    // _backward: Box<dyn FnMut()>,
    // _prev: Vec<Value>,
    // _op: String,
}

impl Value {
    fn new(data: f64) -> Self {
        let mut value = Value { data };
        value
    }
}
//Formating
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.data) // Displaying value with two decimal places
    }
}
//Add
impl<T> Add<T> for Value
where
    T: Into<f64>,
{
    type Output = Self;
    fn add(self, other: T) -> Self::Output {
        Self {
            data: self.data + other.into(),
        }
    }
}
impl Add<&Value> for Value {
    type Output = Self;
    fn add(self, other: &Value) -> Self::Output {
        Self {
            data: self.data + other.data,
        }
    }
}
impl Add for Value {
    type Output = Value;
    fn add(self, other: Self) -> Self::Output {
        Value {
            data: self.data + other.data,
        }
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, other: Self) -> Self::Output {
        Value {
            data: self.data + other.data,
        }
    }
}
impl Add<Value> for &Value {
    type Output = Value;
    fn add(self, other: Value) -> Self::Output {
        Value {
            data: self.data + other.data,
        }
    }
}
impl<T> Add<T> for &Value
where
    T: Into<f64>,
{
    type Output = Value;
    fn add(self, other: T) -> Self::Output {
        Value {
            data: self.data + other.into(),
        }
    }
}

//Implementing multiply
impl<T> Mul<T> for Value
where
    T: Into<f64>,
{
    type Output = Self;
    fn mul(self, other: T) -> Self::Output {
        Self {
            data: self.data * other.into(),
        }
    }
}
impl Mul<&Value> for Value {
    type Output = Self;
    fn mul(self, other: &Value) -> Self::Output {
        Self {
            data: self.data * other.data,
        }
    }
}
impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Self) -> Self::Output {
        Value {
            data: self.data * other.data,
        }
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, other: Self) -> Self::Output {
        Value {
            data: self.data * other.data,
        }
    }
}
impl Mul<Value> for &Value {
    type Output = Value;
    fn mul(self, other: Value) -> Self::Output {
        Value {
            data: self.data * other.data,
        }
    }
}
impl<T> Mul<T> for &Value
where
    T: Into<f64>,
{
    type Output = Value;
    fn mul(self, other: T) -> Self::Output {
        Value {
            data: self.data * other.into(),
        }
    }
}

fn f(x: f64) -> f64 {
    3.00 * f64::powf(x, 2.00) - 4.00 * x + 5.00
}

fn main() {
    let a = Value::new(2.00);
    let b = Value::new(-3.00);
    let c = Value::new(10.00);
    let d = &a * b + c + 1.00;
    // let result = f(x + h);
    // let num = f(x);
    // let diff = (result - num) / h;
    println!("The orginal is {} and the function is {}", a, d)
}
