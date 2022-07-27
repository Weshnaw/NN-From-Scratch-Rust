//! TODO:
//! * ADD doc comments mainly to bnetwork
//! * ADD tests mainly to bnetwork
//! * ADD error handling

/// Contains the functionality to create a neural networks.
mod network;
/// Allows the creation of some random test data for a classification neural network.
mod random_dataset;
mod wlearn;

use crate::network::*;
use crate::random_dataset::*;

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use std::error::Error;
use tracing::*;
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;

use wlearn::layer;
use wlearn::loss;
use wlearn::model::Model;
use wlearn::optimizer;

/// Program entry point
fn main() -> Result<(), Box<dyn Error>> {
  fmt()
    .with_span_events(FmtSpan::CLOSE)
    .with_max_level(Level::DEBUG)
    .init();

  //book()?;
  model()?;

  Ok(())
}

#[instrument(level = "info")]
fn model() -> Result<(), Box<dyn Error>> {
  let mut model = Model::new(2, Box::new(loss::CategoricalCrossEntropy()))
    .add_layer(layer::Dense::new(64))
    .add_layer(layer::ReLUActivation::new())
    .add_layer(layer::Dense::new(64))
    .add_layer(layer::ReLUActivation::new())
    .add_layer(layer::Dense::new(3))
    .add_layer(layer::SoftMaxActivation::new())
    .optimized_model_structure();

  let training_data = Spiral::generate(100, 3);

  let output = model.forward(&training_data.to_ndarray());
  let loss = model.avg_loss(&training_data.class_data());
  let accuracy = class_accuracy(&output, &training_data.class_data());

  info!("Avg Loss: {loss:.2}", loss = loss);
  info!("Accuracy: {acc:.2}", acc = accuracy);
  info!("------------------");

  model.learn(
    &training_data.to_ndarray(),
    &training_data.class_data(),
    optimizer::Sgd::new(1., 1e-3),
    1000,
  );

  let output = model.forward(&training_data.to_ndarray());
  let loss = model.avg_loss(&training_data.class_data());
  let accuracy = class_accuracy(&output, &training_data.class_data());

  info!("Avg Loss: {loss:.2}", loss = loss);
  info!("Accuracy: {acc:.2}", acc = accuracy);
  info!("------------------");

  Ok(())
}

#[instrument(level = "info")]
#[allow(dead_code)]
fn book() -> Result<(), Box<dyn Error>> {
  // Initial Data
  let (x, y) = sine_data(100);

  let mut dense1 = LayerDense::new(1, 64, 0., 0., 0., 0.);
  let mut activation1 = LayerReLUActivation::new();
  let mut dense2 = LayerDense::new(64, 64, 0., 0., 0., 0.);
  let mut activation2 = LayerReLUActivation::new();
  let mut dense3 = LayerDense::new(64, 1, 0., 0., 0., 0.);
  let mut activation3 = LayerLinearActivation::new();
  let mut loss_function = LayerMeanSquareErrorLoss::new();

  let mut optimizer = OptimizerAdam::new(0.005 /*0.0001*/, 1e-3 /*0*/, 1e-7, 0.9, 0.999);

  let learning_span = info_span!("learning").entered();
  let mut thousand_epoch_span = debug_span!(parent: &learning_span, "thousand_epoch").entered();
  debug!("Start learning...");
  let precision = y.std(0.) / 250.;
  for epoch in 1..=10_000 {
    let _epoch_span = trace_span!(parent: &thousand_epoch_span, "epoch", epoch).entered();

    dense1.forward(&Some(x.clone()));
    activation1.forward(&dense1.outputs);
    dense2.forward(&activation1.outputs);
    activation2.forward(&dense2.outputs);
    dense3.forward(&activation2.outputs);
    activation3.forward(&dense3.outputs);
    let regularization_loss =
      dense1.regularization_loss() + dense2.regularization_loss() + dense3.regularization_loss();
    let data_loss = loss_function.forward(&activation3.outputs, &y);
    let loss = data_loss + regularization_loss;
    let outputs = activation3.outputs.as_ref().unwrap();
    trace!("outputs: \n{:?}", outputs);
    let accuracy: f64 = (outputs - &y)
      .mapv(|x| if x.abs() < precision { 1. } else { 0. })
      .mean()
      .unwrap();

    if epoch % 100 == 0 {
      debug!(
        "epoch: {epoch} acc: {acc:.3} loss: {loss:.3} data_loss: {dl:.3} reg_loss: {rl:.3} lr: {lr:.10}",
        lr = optimizer.decayed_rate,
        rl = regularization_loss,
        dl = data_loss,
        loss = loss,
        acc = accuracy,
        epoch = epoch
      );
    } else {
      trace!(
        "epoch: {epoch} acc: {acc:.3} loss: {loss:.3} data_loss: {dl:.3} reg_loss: {rl:.3} lr: {lr:.10}",
        lr = optimizer.decayed_rate,
        rl = regularization_loss,
        dl = data_loss,
        loss = loss,
        acc = accuracy,
        epoch = epoch
      );
    } //loss_activation.backward(&outputs, &training_class_data); // Why are we passing in the outputs? (as loss_activation should already contain the outputs)

    loss_function.backward(outputs, &y);
    activation3.backward(&loss_function.d_inputs);
    dense3.backward(&activation3.d_inputs);
    activation2.backward(&dense3.d_inputs);
    dense2.backward(&activation2.d_inputs);
    activation1.backward(&dense2.d_inputs);
    dense1.backward(&activation1.d_inputs);

    optimizer.pre_update(epoch);
    optimizer.update(&mut dense1);
    optimizer.update(&mut dense2);
    optimizer.update(&mut dense3);
    // TODO: ADD graphing
    // * heat graphs of the normalized weights, biases, and their updates
    // * a line graph of loss, accuracy, and the learning rate
    // * the network's output (each color can be represented by the confidence percent) mapped over the input data graph
    if epoch % 1_000 == 0 {
      thousand_epoch_span = debug_span!(parent: &learning_span, "thousand_epoch").entered();
    }
  }
  learning_span.exit();

  let (x, y) = sine_data(1000);
  dense1.forward(&Some(x));
  activation1.forward(&dense1.outputs);
  dense2.forward(&activation1.outputs);
  activation2.forward(&dense2.outputs);
  dense3.forward(&activation2.outputs);
  activation3.forward(&dense3.outputs);
  let loss = loss_function.forward(&activation3.outputs, &y);
  let outputs = activation3.outputs.as_ref().unwrap();
  trace!("outputs: \n{:?}", outputs);
  let accuracy: f64 = (outputs - &y)
    .mapv(|x| if x.abs() < precision { 1. } else { 0. })
    .mean()
    .unwrap();

  info!(
    "validation, acc: {acc:.3} loss: {loss:.3}",
    acc = accuracy,
    loss = loss
  );

  Ok(())
}

// TODO move into an accuracy module (this would be under class_accuracy::with_label_encoding)
pub fn class_accuracy(predicted: &Array2<f64>, targets: &Array1<usize>) -> f64 {
  predicted
    .outer_iter()
    .zip(targets.iter())
    .fold(0., |total_correct, (confidence, &target)| {
      if confidence.argmax().unwrap() == target {
        total_correct + 1.
      } else {
        total_correct
      }
    })
    / targets.len() as f64
}
