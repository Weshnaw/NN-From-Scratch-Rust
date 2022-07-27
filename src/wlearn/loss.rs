use crate::wlearn::*;
use ndarray::prelude::*;

use super::layer::SoftMaxActivation;

pub struct CategoricalCrossEntropy();

impl CategoricalCrossEntropy {
  fn loss(model_outputs: &MLResult, targets: &Array1<usize>) -> Array1<f64> {
    model_outputs
      .outputs()
      .outer_iter()
      .zip(targets.iter())
      .map(|(confidence, &actual_class)| -confidence[actual_class].clamp(f64::MIN, 1.).ln()) // NOTE: clamping to avoid log(0)
      .collect()
  }
}

impl Loss<usize> for CategoricalCrossEntropy {
  fn loss(&self, model_outputs: &MLResult, targets: &Array1<usize>) -> Array1<f64> {
    Self::loss(model_outputs, targets)
  }

  fn d_loss(&self, model_outputs: &MLResult, targets: &Array1<usize>) -> MLDerivatives {
    let d_values = model_outputs.outputs();
    let samples = d_values.len_of(Axis(0));
    let labels = d_values.len_of(Axis(1));

    let mut one_hot: Array2<f64> = Array::zeros((samples, labels));
    for (mut sample, target) in one_hot.outer_iter_mut().zip(targets.iter()) {
      sample[*target] = 1.0;
    }

    let d_inputs = (-&one_hot / d_values) / samples as f64;

    MLDerivatives {
      d_inputs: Some(d_inputs),
    }
  }

  fn optimize_backward_with_last_layer(
    &self,
    last_layer_type: TypeId,
  ) -> Option<Box<dyn Loss<usize>>> {
    if last_layer_type == TypeId::of::<SoftMaxActivation>() {
      Some(Box::new(SoftMaxCrossEntropy()))
    } else {
      None
    }
  }
}
pub struct SoftMaxCrossEntropy();

impl Loss<usize> for SoftMaxCrossEntropy {
  fn loss(&self, model_outputs: &MLResult, targets: &Array1<usize>) -> Array1<f64> {
    CategoricalCrossEntropy::loss(model_outputs, targets)
  }

  fn d_loss(&self, model_outputs: &MLResult, targets: &Array1<usize>) -> MLDerivatives {
    let d_values = model_outputs.outputs();
    let samples = d_values.len_of(Axis(0));

    let mut d_inputs = d_values.clone();
    for (mut sample, &target) in d_inputs.outer_iter_mut().zip(targets.iter()) {
      sample[target] -= 1.0;
    }

    let d_inputs = d_inputs / samples as f64;

    MLDerivatives {
      d_inputs: Some(d_inputs),
    }
  }

  fn correct_last_layer(&self, last_layer_type: TypeId) -> Option<Box<dyn Layer>> {
    if last_layer_type != TypeId::of::<SoftMaxActivation>() {
      Some(Box::new(SoftMaxActivation::new()))
    } else {
      None
    }
  }
}
