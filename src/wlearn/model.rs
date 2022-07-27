#![allow(dead_code)]

use std::{any::TypeId, ops::Deref};

use crate::wlearn::*;

use tracing::*;

use super::layer::LinearActivation;

pub struct Model<T> {
  layers: Vec<Box<dyn Layer>>, // TODO: Consider adding a type to the layer trait
  loss: Box<dyn Loss<T>>,
  n_inputs: usize,
  skip_last_layer_backward: bool,
}

impl<T> Model<T> {
  #[must_use]
  pub fn new(n_inputs: usize, loss: Box<dyn Loss<T>>) -> Self {
    Self {
      layers: Vec::new(),
      loss,
      n_inputs,
      skip_last_layer_backward: false,
    }
  }

  #[must_use]
  pub fn add_layer(mut self, mut layer: impl Layer) -> Self {
    let layer_inputs = self.get_current_output_size();

    layer.initialize(layer_inputs);
    self.layers.push(Box::new(layer));

    self
  }

  #[must_use]
  fn get_current_output_size(&self) -> usize {
    for layer in self.layers.iter().rev() {
      if let Some(size) = layer.neuron_count() {
        return size;
      }
    }

    self.n_inputs
  }

  #[must_use]
  pub fn correct_potential_mistakes(mut self) -> Self {
    if let Some(last_layer) = self.layers.last() {
      if let Some(mut correct_last_layer) =
        self.loss.correct_last_layer(last_layer.deref().type_id())
      {
        warn!("Correcting potential mistakes in last layer: \n the loss function used should have a LAYER::{:?} preceding it.", last_layer.type_id());
        let layer_inputs = self.get_current_output_size();
        correct_last_layer.initialize(layer_inputs);
        self.layers.push(correct_last_layer);
      }
    }

    self
  }

  #[must_use]
  pub fn optimized_model_structure(mut self) -> Self {
    self = self.correct_potential_mistakes();

    // Optimize: remove linear activations
    self
      .layers
      .retain(|layer| layer.deref().type_id() != TypeId::of::<LinearActivation>());

    // Optimize: attempt to combine loss backward with last layer backward
    if let Some(last_layer) = self.layers.last() {
      if let Some(optimized_loss) = self
        .loss
        .optimize_backward_with_last_layer(last_layer.deref().type_id())
      {
        self.loss = optimized_loss;
        self.skip_last_layer_backward = true;
      }
    }

    self
  }

  #[must_use]
  pub fn should_skip_last_backward(mut self) -> Self {
    self.skip_last_layer_backward = true;
    self
  }

  pub fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
    let inputs = MLResult {
      inputs: None,
      outputs: Some(inputs.clone()),
    };

    let outputs = self.layers.iter_mut().fold(inputs, |layer_inputs, layer| {
      layer.forward(&layer_inputs).to_owned()
    });

    outputs.outputs().to_owned()
  }

  fn last_layer_results(&self) -> MLResult {
    match self.layers.last() {
      Some(layer) => layer.result(),
      None => {
        panic!("No forward pass has been performed");
      }
    }
  }

  pub fn loss(&self, targets: &Array1<T>) -> Array1<f64> {
    let last_layer_result = self.last_layer_results();

    self.loss.loss(&last_layer_result, targets)
  }

  pub fn avg_loss(&self, targets: &Array1<T>) -> f64 {
    let loss = self.loss(targets);

    loss.mean().unwrap()
  }

  pub fn backward(&mut self, targets: &Array1<T>) {
    let last_layer_result = self.last_layer_results();

    let d_loss = self.loss.d_loss(&last_layer_result, targets);

    let mut skip_next_backward = self.skip_last_layer_backward;

    self.layers.iter_mut().rev().fold(d_loss, |d_input, layer| {
      let d_output;

      if !skip_next_backward {
        skip_next_backward = false;
        d_output = layer.backward(&d_input);
      } else {
        d_output = d_input;
      }

      d_output
    });
  }

  pub fn optimize_parameters(&mut self, optimizer: &mut dyn Optimizer) {
    optimizer.update_layers(&mut self.layers);
  }

  pub fn learn(
    &mut self,
    inputs: &Array2<f64>,
    targets: &Array1<T>,
    mut optimizer: impl Optimizer,
    epochs: usize,
  ) {
    optimizer.initialize();

    for epoch in 0..epochs {
      self.forward(inputs);
      self.backward(targets);
      self.optimize_parameters(&mut optimizer);

      if epoch % 100 == 0 {
        debug!(
          "epoch: {epoch} loss: {loss:.5}",
          loss = self.avg_loss(targets),
          epoch = epoch
        );
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_correct_potential_mistakes_soft_max_cross_entropy_without_softmax_activation() {
    let model =
      Model::new(2, Box::new(loss::SoftMaxCrossEntropy())).add_layer(layer::Dense::new(64));

    assert_eq!(model.layers.len(), 1);

    let model = model.correct_potential_mistakes();

    assert_eq!(model.layers.len(), 2);
    assert_eq!(
      model.layers.last().unwrap().deref().type_id(),
      TypeId::of::<layer::SoftMaxActivation>()
    );
  }

  #[test]
  fn test_optimized_model_structure_combine_categorical_cross_entropy_and_softmax() {
    let model = Model::new(2, Box::new(loss::CategoricalCrossEntropy()))
      .add_layer(layer::Dense::new(64))
      .add_layer(layer::SoftMaxActivation::new());

    assert_eq!(model.layers.len(), 2);
    assert!(!model.skip_last_layer_backward);

    let model = model.optimized_model_structure();

    assert_eq!(model.layers.len(), 2);
    assert_eq!(
      model.loss.deref().type_id(),
      TypeId::of::<loss::SoftMaxCrossEntropy>()
    );
    assert!(model.skip_last_layer_backward);
  }

  #[test]
  fn test_optimized_model_structure_remove_linear_activations() {
    let model = Model::new(2, Box::new(loss::CategoricalCrossEntropy()))
      .add_layer(layer::Dense::new(64))
      .add_layer(layer::LinearActivation::new());

    assert_eq!(model.layers.len(), 2);

    let model = model.optimized_model_structure();

    assert_eq!(model.layers.len(), 1);
  }
}
