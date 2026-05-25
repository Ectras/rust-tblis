use std::{marker::PhantomData, mem::MaybeUninit};

use num_complex::Complex64;

/// A view into a tensor with labels, shape, and data.
pub struct TensorView<'a> {
    /// Labels assign a name to each dimension of the tensor.
    labels: &'a [usize],
    /// The shape of the tensor, i.e. the size of each dimension.
    shape: &'a [usize],
    /// The strides of the tensor, i.e. the number of elements to skip to move to the
    /// next element in each dimension.
    strides: &'a [isize],
    /// The data of the tensor.
    data: &'a [Complex64],
}

impl<'a> TensorView<'a> {
    /// Creates a new tensor view.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The number of labels does not match the number of dimensions
    /// - The data length does not match the product of the shape dimensions
    /// - The strides length does not match the number of dimensions
    pub fn new(
        labels: &'a [usize],
        shape: &'a [usize],
        strides: &'a [isize],
        data: &'a [Complex64],
    ) -> Self {
        assert_eq!(labels.len(), shape.len());
        assert_eq!(strides.len(), shape.len());

        let expected_len = shape.iter().product();
        assert_eq!(data.len(), expected_len);

        TensorView {
            labels,
            shape,
            strides,
            data,
        }
    }

    /// Returns the labels of this tensor view.
    #[inline]
    pub fn labels(&self) -> &'a [usize] {
        self.labels
    }

    /// Returns the shape of this tensor view.
    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    /// Returns the data of this tensor view.
    #[inline]
    pub fn data(&self) -> &'a [Complex64] {
        self.data
    }

    /// Returns the strides of this tensor view.
    #[inline]
    pub fn strides(&self) -> &'a [isize] {
        self.strides
    }
}

/// Rust wrapper around an instance of a `tblis_tensor` C struct.
struct TblisTensor<'a> {
    tensor: tblis_sys::tblis_tensor,
    /// Since the tblis_tensor stores pointers to the shape, stride and data, we must
    /// ensure that they outlive this struct.
    phantom: PhantomData<&'a tblis_sys::tblis_tensor>,
}

/// Builds a stride vector for a standard row-major tensor.
fn build_row_major_strides(shape: &[isize]) -> Vec<isize> {
    let mut last = 1;
    let mut strides = vec![0; shape.len()];
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = last;
        last *= dim;
    }
    strides
}

fn create_input_tensor<'a>(
    shape: &'a [isize],
    data: &'a [Complex64],
    strides: &'a [isize],
) -> TblisTensor<'a> {
    let tensor = unsafe {
        let mut tensor = MaybeUninit::<tblis_sys::tblis_tensor>::uninit();
        tblis_sys::tblis_init_tensor_z(
            tensor.as_mut_ptr(),
            shape.len().try_into().unwrap(),
            shape.as_ptr().cast_mut(),
            // tblis' API doesn't discriminate between input and output tensors,
            // hence the data pointer is always passed as non-const and we need cast_mut().
            data.as_ptr().cast_mut().cast(),
            strides.as_ptr().cast_mut(),
        );
        tensor.assume_init()
    };

    TblisTensor {
        tensor,
        phantom: PhantomData,
    }
}

fn create_out_tensor<'a>(
    shape: &'a [isize],
    data: *mut Complex64,
    strides: &'a [isize],
) -> TblisTensor<'a> {
    let tensor = unsafe {
        let mut tensor = MaybeUninit::<tblis_sys::tblis_tensor>::uninit();
        let scalar = tblis_sys::__BindgenComplex::default();
        tblis_sys::tblis_init_tensor_scaled_z(
            tensor.as_mut_ptr(),
            scalar,
            shape.len().try_into().unwrap(),
            shape.as_ptr().cast_mut(),
            data.cast(),
            strides.as_ptr().cast_mut(),
        );
        tensor.assume_init()
    };

    TblisTensor {
        tensor,
        phantom: PhantomData,
    }
}

fn convert_shape(shape: &[usize]) -> Vec<isize> {
    shape.iter().map(|&u| u.try_into().unwrap()).collect()
}

/// Contracts two tensors and returns the output data in the order given by
/// `out_labels`.
pub fn tensor_mult(
    out_labels: &[usize],
    out_shape: &[usize],
    a: TensorView,
    b: TensorView,
) -> Vec<Complex64> {
    let a_shape = convert_shape(a.shape());
    let b_shape = convert_shape(b.shape());
    let out_size = out_shape.iter().product();
    let out_shape = convert_shape(out_shape);
    let out_strides = build_row_major_strides(&out_shape);
    let mut out_data = Vec::with_capacity(out_size);

    let a_tensor = create_input_tensor(&a_shape, a.data(), a.strides());
    let b_tensor = create_input_tensor(&b_shape, b.data(), b.strides());
    let mut c_tensor = create_out_tensor(&out_shape, out_data.as_mut_ptr(), &out_strides);

    unsafe {
        tblis_sys::tblis_tensor_mult(
            std::ptr::null(),
            std::ptr::null(),
            &raw const a_tensor.tensor,
            a.labels().as_ptr(),
            &raw const b_tensor.tensor,
            b.labels().as_ptr(),
            &raw mut c_tensor.tensor,
            out_labels.as_ptr(),
        );
        out_data.set_len(out_size);
    }
    out_data
}

/// Permutes the tensor data to fit the order in `out_labels`.
pub fn tensor_reorder(out_labels: &[usize], out_shape: &[usize], a: TensorView) -> Vec<Complex64> {
    let a_shape = convert_shape(a.shape());
    let out_size = out_shape.iter().product();
    let out_shape = convert_shape(out_shape);
    let out_strides = build_row_major_strides(&out_shape);
    let mut out_data = Vec::with_capacity(out_size);

    let a_tensor = create_input_tensor(&a_shape, a.data(), a.strides());
    let mut c_tensor = create_out_tensor(&out_shape, out_data.as_mut_ptr(), &out_strides);

    unsafe {
        tblis_sys::tblis_tensor_add(
            std::ptr::null(),
            std::ptr::null(),
            &raw const a_tensor.tensor,
            a.labels().as_ptr(),
            &raw mut c_tensor.tensor,
            out_labels.as_ptr(),
        );
        out_data.set_len(out_size);
    }

    out_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_strides() {
        let shape = &[10, 2, 6, 32];
        let strides = build_row_major_strides(shape);
        assert_eq!(strides, vec![384, 192, 32, 1]);
    }

    #[test]
    fn contract_simple() {
        let a_data = &[
            Complex64::new(1.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
        ];
        let b_data = &[
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
        ];
        let c_data = tensor_mult(
            &[0, 1],
            &[2, 3],
            TensorView::new(&[0, 1, 2], &[2, 3, 4], &[12, 4, 1], a_data),
            TensorView::new(&[2], &[4], &[1], b_data),
        );
        assert_eq!(
            c_data,
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(23.0, 0.0),
            ]
        );
    }

    #[test]
    fn contract_transposed_output() {
        let a_data = &[
            Complex64::new(1.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
        ];
        let b_data = &[
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
        ];
        let c_data = tensor_mult(
            &[1, 0],
            &[3, 2],
            TensorView::new(&[0, 1, 2], &[2, 3, 4], &[12, 4, 1], a_data),
            TensorView::new(&[2], &[4], &[1], b_data),
        );
        assert_eq!(
            c_data,
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(23.0, 0.0),
            ]
        );
    }

    #[test]
    fn transpose_matrix() {
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ];
        let out = tensor_reorder(
            &[1, 0],
            &[3, 2],
            TensorView::new(&[0, 1], &[2, 3], &[3, 1], &data),
        );

        assert_eq!(
            out,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(6.0, 0.0),
            ]
        );
    }

    #[test]
    fn contract_scalar_output() {
        let a = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let b = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        let c_data = tensor_mult(
            &[],
            &[],
            TensorView::new(&[0], &[3], &[1], &a),
            TensorView::new(&[0], &[3], &[1], &b),
        );
        assert_eq!(c_data, vec![Complex64::new(7.0, 0.0)]);
    }
}
