#![allow(dead_code)]

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use plotters::prelude::*;
use std::collections::HashSet;

pub fn sine_data(samples: usize) -> (Array2<f64>, Array2<f64>) {
  let x: Array2<f64> = Array::linspace(0., 1., samples)
    .into_shape((samples, 1))
    .unwrap();
  let y: Array2<f64> = x.map(|x| x.sin());
  (x, y)
}

// TODO: make a derive macro for some of the common code such as:
// * to_ndarray()
// * draw_graph()

/// Vertical data that can be used to train a neural network.
pub struct Vertical {
  /// The generated vertical data.
  pub data: Vec<(f64, f64, usize)>,
}

impl Vertical {
  /// Generates a new vertical data set.
  ///
  /// # Arguments
  /// * `points` - The number of points per class.
  /// * `classes` - The number of classes.
  ///
  /// Example:
  /// ```
  /// let vertical = Vertical::generate(100, 3);
  /// assert_eq!(spiral.data.len(), 300);
  /// ```
  pub fn generate(points: usize, classes: usize) -> Self {
    let mut data = vec![];

    for class_number in 0..classes {
      // Adds a bit of noise to the distribution
      let coherence_factor = 0.25; // How close the points are to each other (0.0 being no noise, 1.0 being a lot of noise)
      let randomized_class_distribution = class_number as f64
        + Array::random(points, Normal::new(0.0, 1.0).unwrap()) * coherence_factor;
      let y_factor = Array::linspace(0.0, 1.0, points);
      let xy_points = randomized_class_distribution
        .iter()
        .zip(y_factor.iter())
        .map(|(x, y)| (*x, *y, class_number));
      data.extend(xy_points);
    }

    Vertical { data }
  }

  /// Returns the spiral as a 2D NDArray.
  ///
  /// # Example
  /// ```
  /// let spiral = Spiral::generate_spiral(100, 3);
  /// let nd_array = spiral.as_ndarray();
  /// ```
  pub fn to_ndarray(&self) -> Array2<f64> {
    let mut result = Array2::zeros((self.data.len(), 2));
    for (i, &(x, y, _)) in self.data.iter().enumerate() {
      result[(i, 0)] = x;
      result[(i, 1)] = y;
    }
    result
  }

  /// outputs only the class data
  ///
  /// # Example
  /// ```
  /// let spiral = Spiral::generate_spiral(100, 3);
  /// let class_data = spiral.class_data();
  /// ```
  pub fn class_data(&self) -> Array1<usize> {
    self
      .data
      .iter()
      .map(|&(_, _, class_number)| class_number)
      .collect()
  }

  /// Draws the spiral to a graph.
  ///
  /// # Arguments
  /// * `file` - The file to save the graph to.
  /// * `title` - The title of the graph.
  /// * `show_coords` - A boolean to determine whether or not to show the coordinates of each point.
  ///
  ///
  /// # Example
  /// ```
  ///    let spiral = Spiral::generate(100, 3);
  ///    spiral.draw_graph("test.png", "Test Graph", true).unwrap();
  ///
  ///    assert!(std::path::Path::new("test.png").exists());
  /// ```
  pub fn draw_graph(
    &self,
    file: &str,
    title: &str,
    show_coords: bool,
  ) -> Result<(), Box<dyn std::error::Error>> {
    draw_graph(&self.data, file, title, show_coords, false, false)
  }
}

/// Spiral data that can be used to train a neural network.
pub struct Spiral {
  /// The generated spiral data.
  pub data: Vec<(f64, f64, usize)>,
}

impl Spiral {
  /// Generates spiral data.
  /// Note: having too many classes&points causes many points to overlap.
  ///
  /// # Arguments
  /// * `n` - The number of points to generate per class.
  /// * `classes` - The number of classes to generate.
  ///
  /// # Example
  /// ```
  /// let spiral = Spiral::generate_spiral(100, 3);
  /// assert_eq!(spiral.data.len(), 300);
  /// ```
  pub fn generate(points: usize, classes: usize) -> Self {
    let mut data = vec![];
    for class_number in 0..classes {
      // Adds a bit of separation to each class
      let scalar = 31.4159 / classes as f64;
      let class_distribution_start = class_number as f64 * scalar;
      //println!("class_distribution_start: {}", class_distribution_start);
      let class_distribution = Array::linspace(
        class_distribution_start,
        class_distribution_start + scalar,
        points,
      );
      // Adds a bit of noise to the distribution
      let coherence_factor = 0.5; // How close the points are to each other (0.0 being no noise, 1.0 being a lot of noise)
      let randomized_class_distribution = class_distribution
        + Array::random(points, Normal::new(0.0, 1.0).unwrap()) * coherence_factor;
      // Lets the circle spiral outwards by scaling the x,y values upwards
      let spiral_additive = Array::linspace(0.0, 1.0, points);
      // Actually generates the x,y points based on the cos/sin of the randomized_class_distribution
      // then spirals out the distribution with the spiral_additive
      let sin_distribution = &spiral_additive * randomized_class_distribution.map(|x| x.sin());
      let cos_distribution = &spiral_additive * randomized_class_distribution.map(|x| x.cos());
      // Creates a convenient vector of tuples for graphing and classification
      let xy_points = sin_distribution
        .iter()
        .zip(cos_distribution.iter())
        .map(|(x, y)| (*x, *y, class_number));
      data.extend(xy_points);
    }
    Spiral { data }
  }

  /// Returns the spiral as a 2D NDArray.
  ///
  /// # Example
  /// ```
  /// let spiral = Spiral::generate_spiral(100, 3);
  /// let nd_array = spiral.as_ndarray();
  /// ```
  pub fn to_ndarray(&self) -> Array2<f64> {
    let mut result = Array2::zeros((self.data.len(), 2));
    for (i, &(x, y, _)) in self.data.iter().enumerate() {
      result[(i, 0)] = x;
      result[(i, 1)] = y;
    }
    result
  }

  /// outputs only the class data
  ///
  /// # Example
  /// ```
  /// let spiral = Spiral::generate_spiral(100, 3);
  /// let class_data = spiral.class_data();
  /// ```
  pub fn class_data(&self) -> Array1<usize> {
    self
      .data
      .iter()
      .map(|&(_, _, class_number)| class_number)
      .collect()
  }

  /// Draws the spiral to a graph.
  ///
  /// # Arguments
  /// * `file` - The file to save the graph to.
  /// * `title` - The title of the graph.
  /// * `show_coords` - A boolean to determine whether or not to show the coordinates of each point.
  ///
  ///
  /// # Example
  /// ```
  ///    let spiral = Spiral::generate(100, 3);
  ///    spiral.draw_graph("test.png", "Test Graph", true).unwrap();
  ///
  ///    assert!(std::path::Path::new("test.png").exists());
  /// ```
  pub fn draw_graph(
    &self,
    file: &str,
    title: &str,
    show_coords: bool,
  ) -> Result<(), Box<dyn std::error::Error>> {
    draw_graph(&self.data, file, title, show_coords, true, true)
  }
}

fn draw_graph(
  data: &[(f64, f64, usize)],
  file: &str,
  title: &str,
  show_coords: bool,
  y_symmetry: bool,
  x_symmetry: bool,
) -> Result<(), Box<dyn std::error::Error>> {
  // Initialize the graph
  let root = BitMapBackend::new(file, (640, 480)).into_drawing_area();
  root.fill(&WHITE)?;
  // Get the max x and max y
  let padding_scalar = 1.2;
  let x_upper_bound = data
    .iter()
    .map(|&(x, _, _)| if x_symmetry { x.abs() } else { x })
    .reduce(f64::max)
    .unwrap()
    * padding_scalar;
  let y_upper_bound = data
    .iter()
    .map(|&(_, y, _)| if y_symmetry { y.abs() } else { y })
    .reduce(f64::max)
    .unwrap()
    * padding_scalar;
  // Create the chart
  let y_lower_bound = if y_symmetry {
    -(y_upper_bound * padding_scalar)
  } else {
    data.iter().map(|&(_, y, _)| y).reduce(f64::min).unwrap() * padding_scalar
  };
  let x_lower_bound = if x_symmetry {
    -(x_upper_bound * padding_scalar)
  } else {
    data.iter().map(|&(x, _, _)| x).reduce(f64::min).unwrap() * padding_scalar
  };
  let mut chart = ChartBuilder::on(&root)
    .caption(title, ("sans-serif", 50).into_font())
    .margin(5)
    .x_label_area_size(30)
    .y_label_area_size(30)
    .build_cartesian_2d(x_lower_bound..x_upper_bound, y_lower_bound..y_upper_bound)?;
  // Add grid lines
  chart.configure_mesh().draw()?;

  let colors = data
    .iter()
    .map(|&(_, _, color)| match color {
      0 => RED,
      1 => GREEN,
      2 => BLUE,
      _ => BLACK,
    })
    .collect::<HashSet<_>>();
  for color in colors {
    let points = data
      .iter()
      .filter(|&(_, _, color_number)| match color {
        RED => *color_number == 0,
        GREEN => *color_number == 1,
        BLUE => *color_number == 2,
        _ => true,
      })
      .map(|&(x, y, _)| (x, y))
      .collect::<Vec<_>>();
    // Draw data
    chart.draw_series(PointSeries::of_element(
      points,
      3.,
      ShapeStyle::from(&color).filled(),
      &|coord, size, style| {
        let text = if show_coords {
          Text::new(
            format!("({:.2}, {:.2})", coord.0, coord.1),
            (0, -15),
            ("sans-serif", 12),
          )
        } else {
          Text::new(String::new(), (0, 0), ("sans-serif", 12))
        };
        EmptyElement::at(coord) + Circle::new((0, 0), size, style) + text
      },
    ))?;
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_plot() {
    let spiral = Spiral::generate(100, 3);
    spiral.draw_graph("test.png", "Test Graph", true).unwrap();

    assert!(std::path::Path::new("test.png").exists());

    std::fs::remove_file("test.png").unwrap(); // Unsure if this is necessary
  }

  #[test]
  fn test_spiral() {
    let spiral = Spiral::generate(100, 3);
    assert_eq!(spiral.data.len(), 300);
  }
}
