#![allow(dead_code)]

use crate::bnetwork_old::layer::*;
use crate::bnetwork_old::loss::*;
use crate::bnetwork_old::prelude::*;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

// TODO:
// * Doc comments
// * Consider Adding tests
// * Create a macro for LayerTraits.results amd as_[enum] (i.e. as_layer())
// * Consider using a builder pattern for the layers
// * Add logs from the tracing library

#[doc(hidden)]
type Layer = _Layer;

pub enum Learning {
  RandomShifts,
}
pub struct Network {
  layers: Vec<Layer>,
  //layers: Vec<&'a dyn LayerTraits>,
  loss: Loss,
}

impl Network {
  pub fn new(layers: Vec<crate::bnetwork_old::Layer>, loss: Loss) -> Self {
    let layers: Vec<Layer> = layers.into_iter().map(Layer::from).collect();

    Self { layers, loss }
  }

  /// TODO: consider adding a from<Array2<f64>> method to MLResult, then setup the function to accept a type that implements into<MLResult>
  /// e.x. pub fn forward(&mut self, inputs: &impl into<MLResult>) -> Array2<f64>
  /// or   pub fn forward<T>(&mut self, inputs: &T -> Array2<f64> where T: Into<MLResult>)
  pub fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
    let data = MLResult::new(inputs.clone());

    let result = self
      .layers
      .iter_mut()
      .fold(data, |data, layer| layer.forward(&data));

    result.outputs().clone()
  }

  pub fn backward(&mut self, targets: &Array1<usize>) {
    let last_layer = self.layers.last().unwrap();

    let (derivatives, mut activation_included) =
      self
        .loss
        .optimized_backward(last_layer.results(), targets, last_layer.activation());

    self
      .layers
      .iter_mut()
      .rev()
      .fold(derivatives, |derivatives, layer| {
        let derivative = layer.backward(&derivatives, !activation_included);
        activation_included = false;
        derivative
      });
  }

  pub fn loss(&mut self, targets: &Array1<usize>) -> f64 {
    let results = self.layers.last().unwrap().results();

    self.loss.calculate(results, targets)
  }

  pub fn updated_loss(&mut self, inputs: &Array2<f64>, targets: &Array1<usize>) -> f64 {
    self.forward(inputs);
    self.loss(targets)
  }

  pub fn accuracy(&self, targets: &Array1<usize>) -> f64 {
    let output = self.layers.last().unwrap().results().outputs();

    output.outer_iter().zip(targets.iter()).fold(
      0.,
      |total_correct, (confidence, &actual_class)| {
        if confidence.argmax().unwrap() == actual_class {
          total_correct + 1.
        } else {
          total_correct
        }
      },
    ) / targets.len() as f64
  }

  pub fn learn(
    &mut self,
    learning: &Learning,
    learning_rate: f64,
    iterations: usize,
    inputs: &Array2<f64>,
    targets: &Array1<usize>,
    accuracy_threshold: f64,
  ) {
    match learning {
      Learning::RandomShifts => {
        self.random_shift_learning(
          learning_rate,
          iterations,
          inputs,
          targets,
          accuracy_threshold,
        );
      }
    }
  }

  fn random_shift_learning(
    &mut self,
    learning_rate: f64,
    iterations: usize,
    inputs: &Array2<f64>,
    targets: &Array1<usize>,
    accuracy_threshold: f64,
  ) {
    let mut best_loss = self.updated_loss(inputs, targets);
    let mut best_variables: Vec<_> = self
      .layers
      .iter()
      .map(|layer| {
        let (weights, biases) = layer.parameters();
        (weights.clone(), biases.clone())
      })
      .collect();

    for _ in 0..iterations {
      if self.accuracy(targets) > accuracy_threshold {
        break;
      }

      self
        .layers
        .iter_mut()
        .for_each(|layer| layer.random_shift(learning_rate));

      let loss = self.updated_loss(inputs, targets);

      if loss < best_loss {
        best_variables = self
          .layers
          .iter()
          .map(|layer| {
            let (weights, biases) = layer.parameters();
            (weights.clone(), biases.clone())
          })
          .collect();
        best_loss = loss;
      } else {
        self
          .layers
          .iter_mut()
          .zip(best_variables.iter())
          .for_each(|(layer, (weights, biases))| {
            layer.set_parameters(weights.clone(), biases.clone())
          });
      }
    }
  }

  // TODO: probably remove this
  pub fn print_derivatives(&self) {
    for (i, layer) in self.layers.iter().enumerate() {
      match layer {
        Layer::Dense(layer) => {
          println!(
            "Dense{} d_weights:\n{:?}",
            i,
            layer.derivatives.as_ref().unwrap().d_weights()
          );
          println!(
            "Dense{} d_biases:\n{:?}",
            i,
            layer.derivatives.as_ref().unwrap().d_biases()
          );
        }
      }
    }
  }
}
