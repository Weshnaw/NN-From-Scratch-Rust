pub mod layer;
pub mod loss;
pub mod model;
pub mod optimizer;

use core::any::Any;
use ndarray::prelude::*;
use std::any::TypeId;
use std::collections::HashMap;

#[derive(Default, Clone)]
pub struct MLResult {
  pub inputs: Option<Array2<f64>>,
  // TODO: consider instead of saving the output of the previous layer, we can store a reference to the previous layer
  // this would also allow us to implement neuron count for activation layers
  // maybe consider treating the model like a linked list where each layer holds a reference to the current and the next layer
  pub outputs: Option<Array2<f64>>,
}

impl MLResult {
  pub fn outputs(&self) -> &Array2<f64> {
    self.outputs.as_ref().unwrap()
  }

  pub fn inputs(&self) -> &Array2<f64> {
    self.inputs.as_ref().unwrap()
  }
}

#[derive(Default, Clone)]
pub struct MLDerivatives {
  pub d_inputs: Option<Array2<f64>>,
}

impl MLDerivatives {
  pub fn d_inputs(&self) -> &Array2<f64> {
    self.d_inputs.as_ref().unwrap()
  }
}

#[derive(Default, Clone)]
pub struct MLParameters {
  pub weights: Option<Array2<f64>>,
  pub biases: Option<Array1<f64>>,
  pub d_weights: Option<Array2<f64>>,
  pub d_biases: Option<Array1<f64>>,
  pub weight_cache: HashMap<String, Option<Array2<f64>>>,
  pub bias_cache: Option<Array1<f64>>,
}

impl MLParameters {
  pub fn weights(&self) -> &Array2<f64> {
    self.weights.as_ref().unwrap()
  }

  pub fn biases(&self) -> &Array1<f64> {
    self.biases.as_ref().unwrap()
  }

  pub fn d_weights(&self) -> &Array2<f64> {
    self.d_weights.as_ref().unwrap()
  }

  pub fn d_biases(&self) -> &Array1<f64> {
    self.d_biases.as_ref().unwrap()
  }
}

pub trait Layer: Any {
  fn forward(&mut self, layer_input: &MLResult) -> &MLResult;
  fn backward(&mut self, d_output: &MLDerivatives) -> MLDerivatives;
  fn result(&self) -> MLResult;
  fn initialize(&mut self, _n_inputs: usize) {}
  fn neuron_count(&self) -> Option<usize> {
    None
  }
  fn parameters(&mut self) -> Option<&mut MLParameters> {
    None
  }
}

pub trait Loss<T>: Any {
  fn loss(&self, model_outputs: &MLResult, targets: &Array1<T>) -> Array1<f64>;
  fn d_loss(&self, model_outputs: &MLResult, targets: &Array1<T>) -> MLDerivatives;
  fn optimize_backward_with_last_layer(
    &self,
    _last_layer_type: TypeId,
  ) -> Option<Box<dyn Loss<T>>> {
    None
  }

  fn correct_last_layer(&self, _last_layer_type: TypeId) -> Option<Box<dyn Layer>> {
    None
  }
}

pub trait Optimizer: Any {
  fn initialize(&mut self);
  fn update_layers(&mut self, layer: &mut Vec<Box<dyn Layer>>);
}
