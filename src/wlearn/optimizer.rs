use crate::wlearn::*;

pub struct Sgd {
  learning_rate: f64,
  decay_rate: f64,
  step: usize,
}

impl Sgd {
  pub fn new(learning_rate: f64, decay_rate: f64) -> Self {
    Self {
      learning_rate,
      decay_rate,
      step: 0,
    }
  }
}

impl Optimizer for Sgd {
  fn initialize(&mut self) {
    self.step = 0;
  }
  fn update_layers(&mut self, layer: &mut Vec<Box<dyn Layer>>) {
    let decayed_rate = self.learning_rate * (1. / (1.0 + (self.decay_rate * self.step as f64)));

    for layer in layer.iter_mut() {
      if let Some(mut parameters) = layer.parameters() {
        let updated_weights = parameters.weights() - (decayed_rate * parameters.d_weights());
        let updated_biases = parameters.biases() - (decayed_rate * parameters.d_biases());

        parameters.weights = Some(updated_weights);
        parameters.biases = Some(updated_biases);
      }
    }

    self.step += 1;
  }
}
