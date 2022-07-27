#![allow(dead_code)]

use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Binomial, Normal};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use tracing::*;

#[derive(Debug, Clone, Default)]
pub struct LayerMeanAbsoluteErrorLoss {
  pub inputs: Option<Array2<f64>>,
  pub outputs: Option<Array1<f64>>,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerMeanAbsoluteErrorLoss {
  pub fn new() -> Self {
    LayerMeanAbsoluteErrorLoss {
      ..Default::default()
    }
  }

  pub fn forward(&mut self, inputs: &Option<Array2<f64>>, targets: &Array2<f64>) -> f64 {
    self.inputs = inputs.clone();

    let sample_losses = (targets - inputs.as_ref().unwrap())
      .map(|x| x.abs())
      .mean_axis(Axis(1)) // Book has -1 (means last axis, since we are forcing 2d arrays 1 should be equivalent)
      .unwrap();

    self.outputs = Some(sample_losses);

    self.average()
  }

  pub fn backward(&mut self, dvalues: &Array2<f64>, targets: &Array2<f64>) {
    let samples = dvalues.len_of(Axis(0));
    let labels = dvalues.len_of(Axis(1));

    let dvalues = (targets - dvalues).mapv(|x| match x {
      x if x < 0.0 => -1.0,
      x if x > 0.0 => 1.0,
      _ => 0.0,
    }) / labels as f64;

    self.d_inputs = Some(dvalues / samples as f64);
  }

  pub fn average(&self) -> f64 {
    self.outputs.as_ref().unwrap().mean().unwrap()
  }
}

#[derive(Debug, Clone, Default)]
pub struct LayerMeanSquareErrorLoss {
  pub inputs: Option<Array2<f64>>,
  pub outputs: Option<Array1<f64>>,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerMeanSquareErrorLoss {
  pub fn new() -> Self {
    LayerMeanSquareErrorLoss {
      ..Default::default()
    }
  }

  pub fn forward(&mut self, inputs: &Option<Array2<f64>>, targets: &Array2<f64>) -> f64 {
    self.inputs = inputs.clone();

    let sample_losses = (targets - inputs.as_ref().unwrap())
      .map(|x| x * x)
      .mean_axis(Axis(1)) // Book has -1 (means last axis, since we are forcing 2d arrays 1 should be equivalent)
      .unwrap();

    self.outputs = Some(sample_losses);

    self.average()
  }

  pub fn backward(&mut self, dvalues: &Array2<f64>, targets: &Array2<f64>) {
    let samples = dvalues.len_of(Axis(0)) as f64;
    let labels = dvalues.len_of(Axis(1)) as f64;

    let dvalues = (dvalues - targets) * 2.0 / labels;

    self.d_inputs = Some(&dvalues / samples);
  }

  pub fn average(&self) -> f64 {
    self.outputs.as_ref().unwrap().mean().unwrap()
  }
}

#[derive(Debug, Clone, Default)]
pub struct LayerLinearActivation {
  pub outputs: Option<Array2<f64>>,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerLinearActivation {
  pub fn new() -> LayerLinearActivation {
    LayerLinearActivation {
      ..Default::default()
    }
  }
  pub fn forward(&mut self, inputs: &Option<Array2<f64>>) {
    self.outputs = inputs.clone();
  }
  pub fn backward(&mut self, d_outputs: &Option<Array2<f64>>) {
    self.d_inputs = d_outputs.clone();
  }
}

#[derive(Debug, Clone, Default)]
pub struct LayerDropout {
  dropout_rate: f64,
  inputs: Option<Array2<f64>>,
  binary_mask: Option<Array2<f64>>,
  pub outputs: Option<Array2<f64>>,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerDropout {
  pub fn new(dropout_rate: f64) -> Self {
    Self {
      dropout_rate,
      ..Default::default()
    }
  }

  pub fn forward(&mut self, inputs: &Option<Array2<f64>>) {
    self.inputs = inputs.clone();
    let binary_mask = Array::random(
      inputs.as_ref().unwrap().raw_dim(),
      Binomial::new(1, 1. - self.dropout_rate).unwrap(),
    )
    .map(|&x| x as f64);
    self.outputs = Some(inputs.as_ref().unwrap() * &binary_mask);
    self.binary_mask = Some(binary_mask);
  }

  pub fn backward(&mut self, d_inputs: &Option<Array2<f64>>) {
    self.d_inputs = Some(d_inputs.as_ref().unwrap() * self.binary_mask.as_ref().unwrap());
  }
}

/// Dense layer data containing the weights and biases of a layer
#[derive(Debug, Clone, Default)]
pub struct LayerDense {
  n_inputs: usize,
  n_neurons: usize,
  weight_regularizer_l1: f64,
  bias_regularizer_l1: f64,
  weight_regularizer_l2: f64,
  bias_regularizer_l2: f64,
  /// The weights of the layer.
  pub weights: Array2<f64>,
  /// The biases of the layer.
  pub biases: Array2<f64>,
  //--------------------------------
  inputs: Option<Array2<f64>>,
  pub outputs: Option<Array2<f64>>,
  pub d_weights: Option<Array2<f64>>,
  pub d_biases: Option<Array2<f64>>,
  pub d_inputs: Option<Array2<f64>>,
  pub weight_momentum: Option<Array2<f64>>,
  pub bias_momentum: Option<Array2<f64>>,
  pub weight_cache: Option<Array2<f64>>,
  pub bias_cache: Option<Array2<f64>>,
}

/// Implementation Details for the Dense Layer
///
/// # Methods
/// * `new` - Creates a new Dense Layer.
/// * `forward` - Performs a forward pass through the layer.
impl LayerDense {
  /// Creates a new Dense Layer.
  ///
  /// # Arguments
  /// * `n_inputs` - The size of the input.
  /// * `n_neurons` - The number of neurons (output size as well).
  ///
  /// # Example
  /// ```
  /// let layer = LayerDense::new(2, 3);
  /// ```
  pub fn new(
    n_inputs: usize,
    n_neurons: usize,
    weight_regularizer_l1: f64,
    weight_regularizer_l2: f64,
    bias_regularizer_l1: f64,
    bias_regularizer_l2: f64,
  ) -> Self {
    let weights = Array::random((n_inputs, n_neurons), Normal::new(0.0, 0.1).unwrap());
    let biases = Array::zeros((1, n_neurons));
    Self {
      n_inputs,
      n_neurons,
      weights,
      biases,
      weight_regularizer_l1,
      weight_regularizer_l2,
      bias_regularizer_l1,
      bias_regularizer_l2,
      ..Default::default()
    }
  }

  /// Performs a forward pass through the layer, and applies an activation function.
  ///
  /// # Arguments
  /// * `input` - The input to the layer.
  ///
  /// # Example
  /// ```
  /// // NOTE: there exists some &Array2<f64> of your data called input
  /// let layer = LayerDense::new(2, 3);
  /// let output = layer.forward(&input);
  /// ```
  pub fn forward(&mut self, inputs: &Option<Array2<f64>>) {
    let inputs = inputs.clone().unwrap();
    self.inputs = Some(inputs.clone());

    self.outputs = Some(inputs.dot(&self.weights) + &self.biases);

    debug_assert_eq!(
      self.outputs.as_ref().unwrap().len_of(Axis(1)),
      self.n_neurons
    );
  }

  pub fn backward(&mut self, dvalues: &Option<Array2<f64>>) {
    let dvalues = dvalues.as_ref().unwrap();

    self.d_weights = Some(self.inputs.as_ref().unwrap().t().dot(dvalues));

    let summed_dvalues = dvalues.sum_axis(Axis(0));
    self.d_biases = Some(summed_dvalues.into_shape((1, self.n_neurons)).unwrap());

    if self.weight_regularizer_l1 > 0.0 {
      let dl1 = self.weights.mapv(|x| if x > 0.0 { 1. } else { -1. });
      self.d_weights = Some(self.d_weights.as_ref().unwrap() + &(self.weight_regularizer_l1 * dl1));
    }

    if self.weight_regularizer_l2 > 0.0 {
      self.d_weights =
        Some(self.d_weights.as_ref().unwrap() + &(2. * self.weight_regularizer_l2 * &self.weights));
    }

    if self.bias_regularizer_l1 > 0.0 {
      let dl1 = self.biases.mapv(|x| if x > 0.0 { 1. } else { -1. });
      self.d_biases = Some(self.d_biases.as_ref().unwrap() + &(self.bias_regularizer_l1 * dl1));
    }

    if self.bias_regularizer_l2 > 0.0 {
      self.d_biases =
        Some(self.d_biases.as_ref().unwrap() + &(2. * self.bias_regularizer_l2 * &self.biases));
    }

    self.d_inputs = Some(dvalues.dot(&self.weights.t()));
  }

  /// Randomly shifts the weights and biases.
  ///
  /// # Example
  /// ```
  /// let layer = LayerDense::new(2, 3);
  /// layer.random_shift();
  /// ```
  pub fn random_shift(&mut self) {
    let weights_shape = self.weights.raw_dim();
    let biases_shape = self.biases.raw_dim();

    self.weights = &self.weights + Array::random(weights_shape, Normal::new(0.0, 0.5).unwrap());
    self.biases = &self.biases + Array::random(biases_shape, Normal::new(0.0, 0.5).unwrap());
  }

  pub fn regularization_loss(&mut self) -> f64 {
    let mut loss = 0.0;

    if self.weight_regularizer_l1 > 0.0 {
      loss += self.weight_regularizer_l1 * self.weights.mapv(|x| x.abs()).sum();
    }

    if self.weight_regularizer_l2 > 0.0 {
      loss += self.weight_regularizer_l2 * self.weights.mapv(|x| x.powi(2)).sum();
    }

    if self.weight_regularizer_l1 > 0.0 {
      loss += self.weight_regularizer_l1 * self.biases.mapv(|x| x.abs()).sum();
    }

    if self.bias_regularizer_l2 > 0.0 {
      loss += self.weight_regularizer_l2 * self.biases.mapv(|x| x.powi(2)).sum();
    }

    loss
  }
}

// TODO: Take a hard look at sigmoid and BinaryCrossEntropyLoss because the local result is not the same as the book result.
#[derive(Default)]
pub struct LayerSigmoidActivation {
  pub outputs: Option<Array2<f64>>,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerSigmoidActivation {
  pub fn new() -> Self {
    Self {
      ..Default::default()
    }
  }

  pub fn forward(&mut self, inputs: &Option<Array2<f64>>) {
    self.outputs = Some(inputs.as_ref().unwrap().map(|x| 1.0 / (1.0 + (-x).exp())));
  }

  pub fn backward(&mut self, dvalues: &Option<Array2<f64>>) {
    self.d_inputs = Some(
      dvalues.as_ref().unwrap()
        * (1. - self.outputs.as_ref().unwrap())
        * self.outputs.as_ref().unwrap(),
    );
    trace!("sigmoid d_inputs: \n {:?}", self.d_inputs.as_ref().unwrap());
  }
}

#[derive(Debug, Clone)]
pub struct LayerReLUActivation {
  pub inputs: Option<Array2<f64>>,
  pub outputs: Option<Array2<f64>>,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerReLUActivation {
  pub fn new() -> Self {
    Self {
      inputs: None,
      outputs: None,
      d_inputs: None,
    }
  }

  pub fn forward(&mut self, inputs: &Option<Array2<f64>>) {
    let inputs = inputs.clone().unwrap();

    self.inputs = Some(inputs.clone());
    self.outputs = Some(inputs.mapv(|x| x.max(0.0)));
  }

  pub fn backward(&mut self, dvalues: &Option<Array2<f64>>) {
    let mask = self
      .inputs
      .as_ref()
      .unwrap()
      .mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

    self.d_inputs = Some(&mask * dvalues.as_ref().unwrap());
  }
}

#[derive(Debug, Clone)]
pub struct LayerSoftMaxActivation {
  pub inputs: Option<Array2<f64>>,
  pub outputs: Option<Array2<f64>>,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerSoftMaxActivation {
  pub fn new() -> Self {
    Self {
      inputs: None,
      outputs: None,
      d_inputs: None,
    }
  }

  pub fn forward(&mut self, inputs: &Array2<f64>) {
    self.inputs = Some(inputs.clone());

    let rows = inputs.len_of(Axis(0));
    let maxed_inputs = inputs.map_axis(Axis(1), |x| match x.max() {
      Ok(&max) => max,
      Err(_) => panic!("\n\n{:?}\n\n", inputs),
    });

    let exponentiated_values =
      (inputs - &maxed_inputs.into_shape([rows, 1]).unwrap()).mapv(|x| x.exp());
    let exponentiated_sum = exponentiated_values.map_axis(Axis(1), |x| x.sum());
    self.outputs = Some(&exponentiated_values / &exponentiated_sum.into_shape([rows, 1]).unwrap());
  }

  pub fn backward(&mut self, dvalues: &Array2<f64>) {
    let mut d_inputs: Array2<f64> = Array2::zeros(dvalues.raw_dim());

    for ((output, d_value), mut d_input) in self
      .outputs
      .clone()
      .unwrap()
      .outer_iter()
      .zip(dvalues.outer_iter())
      .zip(d_inputs.outer_iter_mut())
    {
      let output_size = output.len();
      let shaped_output = output.to_shape((output_size, 1)).unwrap();
      let jacobian: Array2<f64> =
        Array2::from_diag(&output) - shaped_output.clone().dot(&shaped_output.t());
      //println!("jacobian: {:?}", jacobian);

      d_input += &jacobian.dot(&d_value);
    }

    self.d_inputs = Some(d_inputs);
  }
}

#[derive(Default)]
pub struct LayerBinaryCrossEntropyLoss {
  pub outputs: Option<Array2<f64>>,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerBinaryCrossEntropyLoss {
  pub fn new() -> Self {
    Self {
      ..Default::default()
    }
  }

  pub fn forward(&mut self, inputs: &Option<Array2<f64>>, targets: &Array2<usize>) -> f64 {
    let targets = targets.mapv(|x| x as f64);
    let samples = inputs.as_ref().unwrap().len_of(Axis(0));

    // TODO: separate out function to be cleaner and more readable
    self.outputs = Some(
      (-(&targets
        * inputs
          .as_ref()
          .unwrap()
          .map(|x| x.clamp(1e-7, 1. - 1e-7).ln())
        + (1. - &targets)
          * inputs
            .as_ref()
            .unwrap()
            .map(|x| (1. - x.clamp(1e-7, 1. - 1e-7)).ln())))
      .mean_axis(Axis(1))
      .unwrap()
      .into_shape((samples, 1))
      .unwrap(),
    );

    trace!("loss: \n{:?}", self.outputs.as_ref().unwrap());

    self.average()
  }

  pub fn backward(&mut self, dvalues: &Array2<f64>, targets: &Array2<usize>) {
    let samples = dvalues.len_of(Axis(0));
    let labels = dvalues.len_of(Axis(1));
    let targets = targets.mapv(|x| x as f64);
    let clipped = dvalues.map(|x| x.clamp(1e-7, 1. - 1e-7));

    self.d_inputs = Some(
      (-(&targets / &clipped - (1. - &targets) / (1. - &clipped)) / labels as f64) / samples as f64,
    );

    trace!("d_inputs: \n{:?}", self.d_inputs.as_ref().unwrap());
  }

  pub fn average(&self) -> f64 {
    self.outputs.as_ref().unwrap().mean().unwrap()
  }
}

#[derive(Debug, Clone)]
pub struct LayerCategoricalCrossEntropyLoss {
  pub inputs: Option<Array2<f64>>,
  pub outputs: Option<Array1<f64>>,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerCategoricalCrossEntropyLoss {
  pub fn new() -> Self {
    Self {
      inputs: None,
      outputs: None,
      d_inputs: None,
    }
  }

  pub fn average(&self) -> f64 {
    self.outputs.as_ref().unwrap().mean().unwrap()
  }

  pub fn forward(&mut self, inputs: &Array2<f64>, targets: &Array1<usize>) {
    self.inputs = Some(inputs.clone());

    let result: Array1<f64> = inputs
      .outer_iter()
      .zip(targets.iter())
      .map(|(confidence, &actual_class)| -confidence[actual_class].clamp(1e-7, 1. - 1e-7).ln()) // NOTE: clamping to avoid log(0)
      .collect();

    self.outputs = Some(result);
  }
  pub fn backward(&mut self, dvalues: &Array2<f64>, targets: &Array1<usize>) {
    let samples = dvalues.len_of(Axis(0));
    let labels = dvalues.len_of(Axis(1));

    let mut one_hot: Array2<f64> = Array::zeros((samples, labels));
    for (mut sample, target) in one_hot.outer_iter_mut().zip(targets.iter()) {
      sample[*target] = 1.0;
    }

    self.d_inputs = Some((-&one_hot / dvalues) / samples as f64);
  }
}

#[derive(Debug, Clone)]
pub struct LayerSoftMaxCrossEntropyLoss {
  pub activation: LayerSoftMaxActivation,
  pub loss: LayerCategoricalCrossEntropyLoss,
  pub d_inputs: Option<Array2<f64>>,
}

impl LayerSoftMaxCrossEntropyLoss {
  pub fn new() -> Self {
    Self {
      activation: LayerSoftMaxActivation::new(),
      loss: LayerCategoricalCrossEntropyLoss::new(),
      d_inputs: None,
    }
  }

  pub fn outputs(&self) -> &Option<Array2<f64>> {
    &self.activation.outputs
  }

  pub fn forward(&mut self, inputs: &Option<Array2<f64>>, targets: &Array1<usize>) -> f64 {
    let inputs = inputs.clone().unwrap();
    self.activation.forward(&inputs);

    self
      .loss
      .forward(&self.activation.outputs.clone().unwrap(), targets);

    self.loss.average()
  }

  pub fn backward(&mut self, dvalues: &Array2<f64>, targets: &Array1<usize>) {
    let samples = dvalues.len_of(Axis(0));

    let mut d_inputs = dvalues.clone();
    for (mut sample, &target) in d_inputs.outer_iter_mut().zip(targets.iter()) {
      sample[target] -= 1.0;
    }

    self.d_inputs = Some(d_inputs / samples as f64);
  }
}

/// Performs an accuracy calculation. outputting the average number of correct guesses.
///
/// # Arguments
/// * `class_targets` - The actual classifications of the data.
///
/// # Example
/// ```
/// // NOTE: there exists some &Array2<f64> of your data called input and some Array1<f64> of the classifications called class_targets
/// let network = Network(vec![
///    LayerDense::new(2, 3, ActivationFunctions::relu),
///    LayerDense::new(3, 3, ActivationFunctions::softmax),
/// ]);
/// network.forward(&input);
///
/// let losses = network.accuracy(&class_targets);
/// ```
pub fn accuracy(output: &Array2<f64>, class_targets: &Array1<usize>) -> f64 {
  output.outer_iter().zip(class_targets.iter()).fold(
    0.,
    |total_correct, (confidence, &actual_class)| {
      if confidence.argmax().unwrap() == actual_class {
        total_correct + 1.
      } else {
        total_correct
      }
    },
  ) / class_targets.len() as f64
}

pub struct OptimizerSGD {
  learning_rate: f64,
  decay_rate: f64,
  pub decayed_rate: f64,
  momentum: f64,
}

impl OptimizerSGD {
  pub fn new(learning_rate: f64, decay_rate: f64, momentum: f64) -> Self {
    Self {
      learning_rate,
      decay_rate,
      decayed_rate: learning_rate,
      momentum,
    }
  }

  pub fn pre_update(&mut self, step: usize) {
    self.decayed_rate = self.learning_rate * (1. / (1.0 + (self.decay_rate * step as f64)));
  }

  pub fn update(&self, layer: &mut LayerDense) {
    layer.weights = &layer.weights - (self.decayed_rate * layer.d_weights.as_ref().unwrap());
    layer.biases = &layer.biases - (self.decayed_rate * layer.d_biases.as_ref().unwrap());
  }

  pub fn update_with_momentum(&self, layer: &mut LayerDense) {
    if layer.weight_momentum.is_none() {
      layer.weight_momentum = Some(Array2::zeros(layer.weights.raw_dim()));
      layer.bias_momentum = Some(Array2::zeros(layer.biases.raw_dim()));
    }

    let weight_update = (self.momentum * layer.weight_momentum.as_ref().unwrap())
      - (self.decayed_rate * layer.d_weights.as_ref().unwrap());
    let bias_update = (self.momentum * layer.bias_momentum.as_ref().unwrap())
      - (self.decayed_rate * layer.d_biases.as_ref().unwrap());

    layer.weights = &layer.weights + &weight_update;
    layer.biases = &layer.biases + &bias_update;
    layer.weight_momentum = Some(weight_update);
    layer.bias_momentum = Some(bias_update);
  }
}

pub struct OptimizerAdaGrad {
  learning_rate: f64,
  decay_rate: f64,
  pub decayed_rate: f64,
  epsilon: f64,
}

impl OptimizerAdaGrad {
  // IDEA: have new take a struct with all the parameters that can be defaulted (could also work with layer initialization)
  pub fn new(learning_rate: f64, decay_rate: f64, epsilon: f64) -> Self {
    Self {
      learning_rate,
      decay_rate,
      decayed_rate: learning_rate,
      epsilon,
    }
  }

  pub fn pre_update(&mut self, step: usize) {
    self.decayed_rate = self.learning_rate * (1. / (1.0 + (self.decay_rate * step as f64)));
  }
  pub fn update(&self, layer: &mut LayerDense) {
    // IDEA: have a hashmap of layer.id (random at layer init) -> (weight_cache, bias_cache)
    //    or have each layer hold a layer.optimizer_utilities: enum. object which holds the caches/momentums
    if layer.weight_cache.is_none() {
      layer.weight_cache = Some(Array2::zeros(layer.weights.raw_dim()));
      layer.bias_cache = Some(Array2::zeros(layer.biases.raw_dim()));
    }

    layer.weight_cache = Some(
      layer.weight_cache.as_ref().unwrap()
        + &(layer.d_weights.as_ref().unwrap() * layer.d_weights.as_ref().unwrap()),
    );

    layer.bias_cache = Some(
      layer.bias_cache.as_ref().unwrap()
        + &(layer.d_biases.as_ref().unwrap() * layer.d_biases.as_ref().unwrap()),
    );

    layer.weights = layer.weights.clone()
      - self.decayed_rate * layer.d_weights.as_ref().unwrap()
        / (&layer.weight_cache.as_ref().unwrap().mapv(|x| x.sqrt()) + self.epsilon);

    layer.biases = layer.biases.clone()
      - self.decayed_rate * layer.d_biases.as_ref().unwrap()
        / (&layer.bias_cache.as_ref().unwrap().mapv(|x| x.sqrt()) + self.epsilon);
  }
}

pub struct OptimizerRMSProp {
  learning_rate: f64,
  decay_rate: f64,
  pub decayed_rate: f64,
  epsilon: f64,
  rho: f64,
}

impl OptimizerRMSProp {
  pub fn new(learning_rate: f64, decay_rate: f64, epsilon: f64, rho: f64) -> Self {
    Self {
      learning_rate,
      decay_rate,
      decayed_rate: learning_rate,
      epsilon,
      rho,
    }
  }

  pub fn pre_update(&mut self, step: usize) {
    self.decayed_rate = self.learning_rate * (1. / (1.0 + (self.decay_rate * step as f64)));
  }
  pub fn update(&self, layer: &mut LayerDense) {
    if layer.weight_cache.is_none() {
      layer.weight_cache = Some(Array2::zeros(layer.weights.raw_dim()));
      layer.bias_cache = Some(Array2::zeros(layer.biases.raw_dim()));
    }

    layer.weight_cache = Some(
      self.rho * layer.weight_cache.as_ref().unwrap()
        + (1.0 - self.rho)
          * (layer.d_weights.as_ref().unwrap() * layer.d_weights.as_ref().unwrap()),
    );

    layer.bias_cache = Some(
      self.rho * layer.bias_cache.as_ref().unwrap()
        + (1.0 - self.rho) * (layer.d_biases.as_ref().unwrap() * layer.d_biases.as_ref().unwrap()),
    );

    layer.weights = layer.weights.clone()
      - self.decayed_rate * layer.d_weights.as_ref().unwrap()
        / (&layer.weight_cache.as_ref().unwrap().mapv(|x| x.sqrt()) + self.epsilon);

    layer.biases = layer.biases.clone()
      - self.decayed_rate * layer.d_biases.as_ref().unwrap()
        / (&layer.bias_cache.as_ref().unwrap().mapv(|x| x.sqrt()) + self.epsilon);
  }
}

pub struct OptimizerAdam {
  learning_rate: f64,
  decay_rate: f64,
  pub decayed_rate: f64,
  epsilon: f64,
  beta_1: f64,
  beta_2: f64,
  step: usize,
}

impl OptimizerAdam {
  pub fn new(learning_rate: f64, decay_rate: f64, epsilon: f64, beta_1: f64, beta_2: f64) -> Self {
    Self {
      learning_rate,
      decay_rate,
      decayed_rate: learning_rate,
      epsilon,
      beta_1,
      beta_2,
      step: 0,
    }
  }

  pub fn pre_update(&mut self, step: usize) {
    self.decayed_rate = self.learning_rate * (1. / (1.0 + (self.decay_rate * step as f64)));
    self.step = step;
  }

  pub fn update(&self, layer: &mut LayerDense) {
    if layer.weight_cache.is_none() {
      layer.weight_momentum = Some(Array2::zeros(layer.weights.raw_dim()));
      layer.bias_momentum = Some(Array2::zeros(layer.biases.raw_dim()));
      layer.weight_cache = Some(Array2::zeros(layer.weights.raw_dim()));
      layer.bias_cache = Some(Array2::zeros(layer.biases.raw_dim()));
    }

    // IDEA: could probably macro these functions, or at the very least use a abstract them out to separate functions
    layer.weight_momentum = Some(
      self.beta_1 * layer.weight_momentum.as_ref().unwrap()
        + (1.0 - self.beta_1) * layer.d_weights.as_ref().unwrap(),
    );

    layer.bias_momentum = Some(
      self.beta_1 * layer.bias_momentum.as_ref().unwrap()
        + (1.0 - self.beta_1) * layer.d_biases.as_ref().unwrap(),
    );

    let weight_momentum_corrected =
      layer.weight_momentum.as_ref().unwrap() / (1.0 - self.beta_1.powi(self.step as i32 + 1));
    let bias_momentum_corrected =
      layer.bias_momentum.as_ref().unwrap() / (1.0 - self.beta_1.powi(self.step as i32 + 1));

    layer.weight_cache = Some(
      self.beta_2 * layer.weight_cache.as_ref().unwrap()
        + (1.0 - self.beta_2)
          * (layer.d_weights.as_ref().unwrap() * layer.d_weights.as_ref().unwrap()),
    );

    layer.bias_cache = Some(
      self.beta_2 * layer.bias_cache.as_ref().unwrap()
        + (1.0 - self.beta_2)
          * (layer.d_biases.as_ref().unwrap() * layer.d_biases.as_ref().unwrap()),
    );

    let weight_cache_corrected =
      layer.weight_cache.as_ref().unwrap() / (1.0 - self.beta_2.powi(self.step as i32 + 1));
    let bias_cache_corrected =
      layer.bias_cache.as_ref().unwrap() / (1.0 - self.beta_2.powi(self.step as i32 + 1));

    layer.weights = layer.weights.clone()
      - self.decayed_rate * weight_momentum_corrected
        / (&weight_cache_corrected.mapv(|x| x.sqrt()) + self.epsilon);
    layer.biases = layer.biases.clone()
      - self.decayed_rate * bias_momentum_corrected
        / (&bias_cache_corrected.mapv(|x| x.sqrt()) + self.epsilon);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_relu_backward() {
    let input = array![[-1., 2., -3.], [4., -5., 6.]];
    let d_value = array![[10., 20., 30.], [40., 50., 60.]];
    let mut relu = LayerReLUActivation::new();
    relu.forward(&Some(input));
    relu.backward(&Some(d_value));

    assert_eq!(relu.inputs.unwrap(), array![[-1., 2., -3.], [4., -5., 6.]]);
    assert_eq!(relu.outputs.unwrap(), array![[0., 2., 0.], [4., 0., 6.]]);
    assert_eq!(
      relu.d_inputs.unwrap(),
      array![[0., 20., 0.], [40., 0., 60.]]
    );
  }

  #[test]
  fn test_cce_loss_forward() {
    let input = array![[0.5, 0.3, 0.2], [0.8, 0.1, 0.1]];

    let mut cce = LayerCategoricalCrossEntropyLoss::new();
    cce.forward(&input, &array![1, 0]);

    //println!("{:?}", cce.outputs.unwrap());
  }

  #[test]
  fn test_smcel_forward() {
    let input = array![[10., 200., 0.], [0.8, 5., 20.]];

    let mut smcel = LayerSoftMaxCrossEntropyLoss::new();

    let _loss = smcel.forward(&Some(input), &array![1, 2]);

    //println!("{:?}", smcel.activation.outputs.unwrap());
    //println!("{:?}", smcel.loss.outputs.unwrap());
    //println!("{:?}", _loss);
  }

  #[test]
  fn test_smcel_backward() {
    let input = array![[10., 200., 0.], [0.8, 5., 20.]];

    let mut smcel = LayerSoftMaxCrossEntropyLoss::new();

    let _loss = smcel.forward(&Some(input.clone()), &array![1, 2]);
    smcel.backward(&input, &array![1, 2]);

    println!("{:?}", smcel.d_inputs);
  }

  #[test]
  fn test_softmax_activation_backward() {
    let input = array![[0.5, 0.3, 0.2]];
    let mut sma = LayerSoftMaxActivation::new();
    sma.forward(&input);
    sma.backward(&input);
  }
}
