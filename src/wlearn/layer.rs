#![allow(dead_code)]

use crate::wlearn::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;

/* Connected Layers */

#[derive(Default)]
pub struct Dense {
  n_neurons: usize,
  weight_regularizer_l1: f64,
  bias_regularizer_l1: f64,
  weight_regularizer_l2: f64,
  bias_regularizer_l2: f64,
  results: Option<MLResult>,
  parameters: MLParameters,
}

impl Dense {
  pub fn new(n_neurons: usize) -> Self {
    Self {
      n_neurons,
      ..Default::default()
    }
  }

  pub fn with_l1(mut self, weight_regularizer_l1: f64, bias_regularizer_l1: f64) -> Self {
    debug_assert!(weight_regularizer_l1 >= 0.0);
    debug_assert!(bias_regularizer_l1 >= 0.0);

    self.weight_regularizer_l1 = weight_regularizer_l1;
    self.bias_regularizer_l1 = bias_regularizer_l1;

    self
  }

  pub fn with_l2(mut self, weight_regularizer_l2: f64, bias_regularizer_l2: f64) -> Self {
    debug_assert!(weight_regularizer_l2 >= 0.0);
    debug_assert!(bias_regularizer_l2 >= 0.0);

    self.weight_regularizer_l2 = weight_regularizer_l2;
    self.bias_regularizer_l2 = bias_regularizer_l2;

    self
  }

  fn weights(&self) -> &Array2<f64> {
    self.parameters.weights()
  }

  fn biases(&self) -> &Array1<f64> {
    self.parameters.biases()
  }

  fn inputs(&self) -> &Array2<f64> {
    self.results.as_ref().unwrap().inputs()
  }
}

impl Layer for Dense {
  fn forward(&mut self, layer_input: &MLResult) -> &MLResult {
    let inputs = layer_input.outputs().to_owned();
    debug_assert_eq!(
      inputs.shape()[1],
      self.parameters.weights().shape()[0],
      "input shape does not fit weight shape"
    );

    let output = &inputs.dot(self.weights()) + self.biases();

    let result = MLResult {
      inputs: Some(inputs),
      outputs: Some(output),
    };

    self.results = Some(result);

    self.results.as_ref().unwrap()
  }

  fn backward(&mut self, d_outputs: &MLDerivatives) -> MLDerivatives {
    let d_outputs = d_outputs.d_inputs();
    // TODO: assert d_outputs shape
    let inputs = self.inputs();

    let mut d_weights = inputs.t().dot(d_outputs);
    let mut d_biases = d_outputs.sum_axis(Axis(0));
    let d_inputs = d_outputs.dot(&self.weights().t());

    if self.weight_regularizer_l1 > 0.0 {
      let dl1 = self.weights().mapv(|x| if x > 0.0 { 1. } else { -1. });
      d_weights = &d_weights + &(&dl1 * self.weight_regularizer_l1);
    }

    if self.bias_regularizer_l1 > 0.0 {
      let dl1 = self.biases().mapv(|x| if x > 0.0 { 1. } else { -1. });
      d_biases = &d_biases + &(&dl1 * self.bias_regularizer_l1);
    }

    if self.weight_regularizer_l2 > 0.0 {
      d_weights = &d_weights + &(2. * self.weight_regularizer_l2 * self.weights());
    }

    if self.bias_regularizer_l2 > 0.0 {
      d_biases = &d_biases + &(2. * self.bias_regularizer_l2 * self.biases());
    }

    self.parameters.d_weights = Some(d_weights);
    self.parameters.d_biases = Some(d_biases);

    MLDerivatives {
      d_inputs: Some(d_inputs),
    }
  }

  fn initialize(&mut self, n_inputs: usize) {
    let weights: Array2<f64> =
      Array::random((n_inputs, self.n_neurons), Normal::new(0.0, 0.1).unwrap());
    let biases: Array1<f64> = Array::zeros(self.n_neurons);

    self.parameters.weights = Some(weights);
    self.parameters.biases = Some(biases);
  }

  fn neuron_count(&self) -> Option<usize> {
    Some(self.n_neurons)
  }

  fn result(&self) -> MLResult {
    self.results.clone().unwrap()
  }

  fn parameters(&mut self) -> Option<&mut MLParameters> {
    Some(&mut self.parameters)
  }
}

/* Activation Layers */

#[derive(Default)]
pub struct ReLUActivation {
  results: Option<MLResult>,
}

impl ReLUActivation {
  pub fn new() -> Self {
    Self {
      ..Default::default()
    }
  }
}

impl Layer for ReLUActivation {
  fn forward(&mut self, layer_input: &MLResult) -> &MLResult {
    let calculation = layer_input.outputs().mapv(|x| x.max(0.0));

    let result = MLResult {
      inputs: layer_input.outputs.clone(),
      outputs: Some(calculation),
    };

    self.results = Some(result);

    self.results.as_ref().unwrap()
  }

  fn backward(&mut self, d_output: &MLDerivatives) -> MLDerivatives {
    let d_output = d_output.d_inputs();
    let inputs = self.results.as_ref().unwrap().inputs();

    let mask = inputs.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

    let d_inputs = &mask * d_output;

    MLDerivatives {
      d_inputs: Some(d_inputs),
    }
  }

  fn result(&self) -> MLResult {
    self.results.clone().unwrap()
  }
}

#[derive(Default)]
pub struct SoftMaxActivation {
  results: Option<MLResult>,
}

impl SoftMaxActivation {
  pub fn new() -> Self {
    Self {
      ..Default::default()
    }
  }
}

impl Layer for SoftMaxActivation {
  fn forward(&mut self, layer_input: &MLResult) -> &MLResult {
    let inputs = layer_input.outputs();
    let rows = inputs.len_of(Axis(0));
    let maxed_inputs = inputs.map_axis(Axis(1), |x| *x.max().unwrap());
    let exponentiated_values =
      (inputs - &maxed_inputs.into_shape([rows, 1]).unwrap()).map(|x| x.exp());
    let exponentiated_sum = exponentiated_values.map_axis(Axis(1), |x| x.sum());
    let calculation = &exponentiated_values / &exponentiated_sum.into_shape([rows, 1]).unwrap();

    let result = MLResult {
      inputs: layer_input.outputs.clone(),
      outputs: Some(calculation),
    };

    self.results = Some(result);

    self.results.as_ref().unwrap()
  }

  fn backward(&mut self, d_output: &MLDerivatives) -> MLDerivatives {
    let d_values = d_output.d_inputs();
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
    }
  }

  fn result(&self) -> MLResult {
    self.results.clone().unwrap()
  }
}

#[derive(Default)]
pub struct LinearActivation {
  results: Option<MLResult>,
}

impl LinearActivation {
  pub fn new() -> Self {
    Self {
      ..Default::default()
    }
  }
}

impl Layer for LinearActivation {
  fn forward(&mut self, layer_input: &MLResult) -> &MLResult {
    let result = MLResult {
      inputs: layer_input.outputs.clone(),
      outputs: layer_input.outputs.clone(),
    };

    self.results = Some(result);

    self.results.as_ref().unwrap()
  }

  fn backward(&mut self, d_output: &MLDerivatives) -> MLDerivatives {
    MLDerivatives {
      d_inputs: Some(d_output.d_inputs().clone()),
    }
  }

  fn result(&self) -> MLResult {
    self.results.clone().unwrap()
  }
}
