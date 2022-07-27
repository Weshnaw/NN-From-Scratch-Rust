#![allow(dead_code)]

use crate::bnetwork_old::prelude::*;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

pub(in crate::bnetwork_old) trait ActivationTraits {
  fn forward(&mut self, data: &MLResult) -> MLResult;
  fn backward(&mut self, data: &MLDerivatives) -> MLDerivatives;
}

#[doc(hidden)]
macro_rules! make_activations {
  ($($(#[$meta:meta])* $x:ident),*) => {
    pub enum Activation {
      $(
          $(#[$meta])*
          $x,
      )*
    }

    #[doc(hidden)]
    pub(in crate::bnetwork_old) enum _Activation {
          $(
              $x($x),
          )*
      }

      impl ActivationTraits for _Activation {
          fn forward(&mut self, data: &MLResult) -> MLResult {
              match self {
                  $(
                    Self::$x(layer) => layer.forward(data),
                  )*
              }
          }
          fn backward(&mut self, data: &MLDerivatives) -> MLDerivatives {
            match self {
                $(
                  Self::$x(layer) => layer.backward(data),
                )*
            }
          }
      }

      impl From<Activation> for _Activation {
        fn from(activation: Activation) -> Self {
          match activation {
            $(
              Activation::$x => _Activation::$x($x::default()),
            )*
          }
        }
      }
  };
}

make_activations!(
  /// Rectified Linear Unit activation
  ReLU,
  /// SoftMax activation
  SoftMax,
  /// Linear activation
  Linear
);

#[derive(Default)]
pub struct Linear();

impl ActivationTraits for Linear {
  fn forward(&mut self, data: &MLResult) -> MLResult {
    MLResult {
      inputs: data.outputs.clone(),
      outputs: data.outputs.clone(),
    }
  }
  fn backward(&mut self, data: &MLDerivatives) -> MLDerivatives {
    data.clone()
  }
}

#[derive(Default)]
pub struct ReLU {
  results: Option<MLResult>,
}

impl ActivationTraits for ReLU {
  fn forward(&mut self, data: &MLResult) -> MLResult {
    let calculation = data.outputs().mapv(|x| x.max(0.0));

    let result = MLResult {
      inputs: data.outputs.clone(),
      outputs: Some(calculation),
    };

    self.results = Some(result.clone());

    result
  }

  fn backward(&mut self, data: &MLDerivatives) -> MLDerivatives {
    let d_values = data.d_inputs();
    let inputs = self.results.as_ref().unwrap().inputs();

    let mask = inputs.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

    let d_inputs = &mask * d_values;

    MLDerivatives {
      d_inputs: Some(d_inputs),
      ..Default::default()
    }
  }
}

#[derive(Default)]
pub struct SoftMax {
  results: Option<MLResult>,
}

impl ActivationTraits for SoftMax {
  fn forward(&mut self, data: &MLResult) -> MLResult {
    let inputs = data.outputs();
    let rows = inputs.len_of(Axis(0));
    let maxed_inputs = inputs.map_axis(Axis(1), |x| *x.max().unwrap());
    let exponentiated_values =
      (inputs - &maxed_inputs.into_shape([rows, 1]).unwrap()).map(|x| x.exp());
    let exponentiated_sum = exponentiated_values.map_axis(Axis(1), |x| x.sum());
    let calculation = &exponentiated_values / &exponentiated_sum.into_shape([rows, 1]).unwrap();

    let result = MLResult {
      inputs: data.outputs.clone(),
      outputs: Some(calculation),
    };

    self.results = Some(result.clone());

    result
  }

  fn backward(&mut self, data: &MLDerivatives) -> MLDerivatives {
    let d_values = data.d_inputs();
    let outputs = self.results.as_ref().unwrap().outputs();

    let mut d_inputs: Array2<f64> = Array2::zeros(d_values.raw_dim());

    for ((output, d_value), mut d_input) in outputs
      .outer_iter()
      .zip(d_values.outer_iter())
      .zip(d_inputs.outer_iter_mut())
    {
      let output_size = output.len();
      let shaped_output = output.to_shape((output_size, 1)).unwrap();
      let jacobian: Array2<f64> =
        Array2::from_diag(&output) - shaped_output.clone().dot(&shaped_output.t());

      d_input += &jacobian.dot(&d_value);
    }

    MLDerivatives {
      d_inputs: Some(d_inputs),
      ..Default::default()
    }
  }
}
