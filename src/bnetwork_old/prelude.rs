use ndarray::prelude::*;

#[derive(Default, Clone)]
pub struct MLResult {
  pub(in crate::bnetwork_old) inputs: Option<Array2<f64>>,
  pub(in crate::bnetwork_old) outputs: Option<Array2<f64>>,
}

impl MLResult {
  pub fn new(initial_data: Array2<f64>) -> Self {
    Self {
      outputs: Some(initial_data),
      ..Default::default()
    }
  }

  pub fn outputs(&self) -> &Array2<f64> {
    self.outputs.as_ref().unwrap()
  }

  pub fn inputs(&self) -> &Array2<f64> {
    self.inputs.as_ref().unwrap()
  }
}

#[derive(Default, Clone)]
pub struct MLDerivatives {
  pub(in crate::bnetwork_old) d_inputs: Option<Array2<f64>>,
  pub(in crate::bnetwork_old) d_weights: Option<Array2<f64>>,
  pub(in crate::bnetwork_old) d_biases: Option<Array1<f64>>,
}

impl MLDerivatives {
  pub fn d_weights(&self) -> &Array2<f64> {
    self.d_weights.as_ref().unwrap()
  }

  pub fn d_biases(&self) -> &Array1<f64> {
    self.d_biases.as_ref().unwrap()
  }

  pub fn d_inputs(&self) -> &Array2<f64> {
    self.d_inputs.as_ref().unwrap()
  }
}
