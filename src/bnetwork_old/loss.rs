#![allow(dead_code)]

use crate::bnetwork_old::activation::*;
use crate::bnetwork_old::prelude::*;
use ndarray::prelude::*;

pub(in crate::bnetwork_old) trait LossTraits {
  fn forward(data: &MLResult, targets: &Array1<usize>) -> Array1<f64>;
  fn backward(data: &MLResult, targets: &Array1<usize>) -> MLDerivatives;
}

#[doc(hidden)]
type Activation = _Activation;

#[doc(hidden)]
macro_rules! make_loss {
  ($($(#[$meta:meta])* $x:ident),*) => {
      pub enum Loss {
          $(
              $(#[$meta])*
              $x,
          )*
      }

      impl Loss {
          fn forward(&self, data: &MLResult, targets: &Array1<usize>) -> Array1<f64> {
              match self {
                  $(
                    Self::$x => $x::forward(data, targets),
                  )*
              }
          }
          fn backward(&self, data: &MLResult, targets: &Array1<usize>) -> MLDerivatives {
            match self {
                $(
                  Self::$x => $x::backward(data, targets),
                )*
            }
          }
      }
  };
}
make_loss!(CategoricalCrossEntropy);

impl Loss {
  pub(in crate::bnetwork_old) fn calculate(&self, data: &MLResult, targets: &Array1<usize>) -> f64 {
    self.forward(data, targets).mean().unwrap()
  }

  pub(in crate::bnetwork_old) fn optimized_backward(
    &self,
    data: &MLResult,
    targets: &Array1<usize>,
    last_activation: &Activation,
  ) -> (MLDerivatives, bool) {
    match self {
      Loss::CategoricalCrossEntropy => match last_activation {
        Activation::SoftMax(_) => (
          CategoricalCrossEntropy::softmax_backward(data, targets),
          true,
        ),
        _ => (self.backward(data, targets), false),
      },
    }
  }
}

pub struct CategoricalCrossEntropy();

impl CategoricalCrossEntropy {
  fn softmax_backward(data: &MLResult, targets: &Array1<usize>) -> MLDerivatives {
    let d_values = data.outputs();
    let samples = d_values.len_of(Axis(0));

    let mut d_inputs = d_values.clone();
    for (mut sample, &target) in d_inputs.outer_iter_mut().zip(targets.iter()) {
      sample[target] -= 1.0;
    }

    let d_inputs = d_inputs / samples as f64;

    MLDerivatives {
      d_inputs: Some(d_inputs),
      d_weights: None,
      d_biases: None,
    }
  }
}

impl LossTraits for CategoricalCrossEntropy {
  fn forward(data: &MLResult, targets: &Array1<usize>) -> Array1<f64> {
    data
      .outputs()
      .outer_iter()
      .zip(targets.iter())
      .map(|(confidence, &actual_class)| -confidence[actual_class].clamp(f64::MIN, 1.).ln()) // NOTE: clamping to avoid log(0)
      .collect()
  }

  fn backward(data: &MLResult, targets: &Array1<usize>) -> MLDerivatives {
    let d_values = data.outputs();
    let samples = d_values.len_of(Axis(0));
    let labels = d_values.len_of(Axis(1));

    let mut one_hot: Array2<f64> = Array::zeros((samples, labels));
    for (mut sample, target) in one_hot.outer_iter_mut().zip(targets.iter()) {
      sample[*target] = 1.0;
    }

    let d_inputs = (-&one_hot / d_values) / samples as f64;

    MLDerivatives {
      d_inputs: Some(d_inputs),
      d_weights: None,
      d_biases: None,
    }
  }
}
