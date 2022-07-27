#![allow(dead_code)]

use crate::bnetwork_old::activation::*;
use crate::bnetwork_old::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

#[doc(hidden)]
type Activation = _Activation;

pub(in crate::bnetwork_old) trait LayerTraits {
  fn forward(&mut self, data: &MLResult) -> MLResult;
  fn backward(&mut self, data: &MLDerivatives, with_activation: bool) -> MLDerivatives;
  fn results(&self) -> &MLResult;
  fn activation(&self) -> &Activation;
  fn parameters(&self) -> (&Array2<f64>, &Array1<f64>);
  fn set_parameters(&mut self, weights: Array2<f64>, biases: Array1<f64>);
  fn random_shift(&mut self, rate: f64);
}

#[doc(hidden)]
macro_rules! make_layers {
  ($($(#[$meta:meta])* $x:ident),*) => {
      /// TODO: consider making the layer macro able to define the layer enum type
      /// i.x. make_layers!(/// Doc comment
      ///                   Dense(
      ///                   /// Doc Comment saying this is n_inputs
      ///                   usize,
      ///                   /// Doc Comment saying this is n_neurons
      ///                   usize,
      ///                   Activation))
      pub enum Layer {
        $(
            $(#[$meta])*
            $x(usize, usize, crate::bnetwork_old::Activation),
        )*
      }

      #[doc(hidden)]
      pub(in crate::bnetwork_old) enum _Layer {
          $(
              $x($x),
          )*
      }

      impl LayerTraits for _Layer {
          fn forward(&mut self, data: &MLResult) -> MLResult {
              match self {
                  $(
                    Self::$x(layer) => layer.forward(data),
                  )*
              }
          }
          fn backward(&mut self, data: &MLDerivatives, with_activation: bool) -> MLDerivatives {
            match self {
                $(
                  Self::$x(layer) => layer.backward(data, with_activation),
                )*
            }
          }
          fn results(&self) -> &MLResult {
            match self {
                $(
                  Self::$x(layer) => layer.results(),
                )*
            }
          }
          fn activation(&self) -> &Activation {
            match self {
                $(
                  Self::$x(layer) => layer.activation(),
                )*
            }
          }
          fn parameters(&self) -> (&Array2<f64>, &Array1<f64>) {
            match self {
                $(
                  Self::$x(layer) => layer.parameters(),
                )*
            }
          }
          fn set_parameters(&mut self, weights: Array2<f64>, biases: Array1<f64>) {
            match self {
                $(
                  Self::$x(layer) => layer.set_parameters(weights, biases),
                )*
            }
          }
          fn random_shift(&mut self, rate: f64) {
            match self {
                $(
                  Self::$x(layer) => layer.random_shift(rate),
                )*
            }
          }
      }

      impl From<Layer> for _Layer {
        fn from(layer: Layer) -> Self {
          match layer {
            $(
              Layer::$x(input_size, output_size, activation) => _Layer::$x($x::new(input_size, output_size, activation)),
            )*
          }
        }
      }
  };
}
make_layers!(
  /// Dense Layer
  ///
  /// # Arguments
  /// Dense(input_size, output_size, activation)
  Dense
);

pub struct Dense {
  weights: Array2<f64>,
  biases: Array1<f64>,
  activation: Activation,
  results: Option<MLResult>,
  pub(in crate::bnetwork_old) derivatives: Option<MLDerivatives>,
}

impl Dense {
  pub fn new(
    n_inputs: usize,
    n_neurons: usize,
    activation: crate::bnetwork_old::Activation,
  ) -> Self {
    let weights = Array::random((n_inputs, n_neurons), Normal::new(0.0, 0.01).unwrap());
    let biases = Array1::zeros(n_neurons);
    let activation = activation.into();

    Self {
      weights,
      biases,
      activation,
      results: None,
      derivatives: None,
    }
  }
}

impl LayerTraits for Dense {
  fn forward(&mut self, data: &MLResult) -> MLResult {
    let dense_calculation = data.outputs().dot(&self.weights) + &self.biases;
    let layer_output = MLResult::new(dense_calculation);
    let activation = self.activation.forward(&layer_output);

    let result = MLResult {
      inputs: data.outputs.clone(),
      outputs: activation.outputs,
    };

    self.results = Some(result.clone());

    result
  }

  fn backward(&mut self, data: &MLDerivatives, with_activation: bool) -> MLDerivatives {
    let activation_derivatives = if with_activation {
      self.activation.backward(data)
    } else {
      data.clone()
    };

    let d_values = activation_derivatives.d_inputs();
    let inputs = self.results.as_ref().unwrap().inputs();

    let d_weights = inputs.t().dot(d_values);
    let d_biases = d_values.sum_axis(Axis(0));
    let d_inputs = d_values.dot(&self.weights.t());

    let result = MLDerivatives {
      d_inputs: Some(d_inputs),
      d_weights: Some(d_weights),
      d_biases: Some(d_biases),
    };

    self.derivatives = Some(result.clone());

    result
  }

  fn results(&self) -> &MLResult {
    self.results.as_ref().unwrap()
  }

  fn activation(&self) -> &Activation {
    &self.activation
  }

  fn parameters(&self) -> (&Array2<f64>, &Array1<f64>) {
    (&self.weights, &self.biases)
  }
  fn set_parameters(&mut self, weights: Array2<f64>, biases: Array1<f64>) {
    self.weights = weights;
    self.biases = biases;
  }
  fn random_shift(&mut self, rate: f64) {
    let weights_shape = self.weights.raw_dim();
    let biases_shape = self.biases.raw_dim();

    self.weights = &self.weights + Array::random(weights_shape, Normal::new(0., rate).unwrap());
    self.biases = &self.biases + Array::random(biases_shape, Normal::new(0., rate).unwrap());
  }
}
