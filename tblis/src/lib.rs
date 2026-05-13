use std::{marker::PhantomData, mem::MaybeUninit};

use num_complex::Complex64;

struct TblisTensor<'a> {
    tensor: tblis_sys::tblis_tensor,
    phantom: PhantomData<&'a tblis_sys::tblis_tensor>,
}

fn build_row_major_strides(shape: &[isize]) -> Vec<isize> {
    let mut last = 1;
    let mut strides = vec![0; shape.len()];
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = last;
        last *= dim;
    }
    strides
}

fn create_tensor<'a>(
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
    data: *const Complex64,
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
            data.cast_mut().cast(),
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

pub fn tensor_mult(
    out_labels: &[usize],
    out_shape: &[usize],
    a_labels: &[usize],
    a_shape: &[usize],
    a_data: &[Complex64],
    b_labels: &[usize],
    b_shape: &[usize],
    b_data: &[Complex64],
) -> Vec<Complex64> {
    let a_shape = convert_shape(a_shape);
    let a_strides = build_row_major_strides(&a_shape);
    let b_shape = convert_shape(b_shape);
    let b_strides = build_row_major_strides(&b_shape);
    let out_size = out_shape.iter().product();
    let out_shape = convert_shape(out_shape);
    let out_strides = build_row_major_strides(&out_shape);
    let mut out_data = Vec::with_capacity(out_size);

    let a_tensor = create_tensor(&a_shape, a_data, &a_strides);
    let b_tensor = create_tensor(&b_shape, b_data, &b_strides);
    let mut c_tensor = create_out_tensor(&out_shape, out_data.as_ptr(), &out_strides);

    unsafe {
        tblis_sys::tblis_tensor_mult(
            std::ptr::null(),
            std::ptr::null(),
            &raw const a_tensor.tensor,
            a_labels.as_ptr(),
            &raw const b_tensor.tensor,
            b_labels.as_ptr(),
            &raw mut c_tensor.tensor,
            out_labels.as_ptr(),
        );
        out_data.set_len(out_size);
    }
    out_data
}

pub fn tensor_reorder(
    out_labels: &[usize],
    out_shape: &[usize],
    a_labels: &[usize],
    a_shape: &[usize],
    a_data: &[Complex64],
) -> Vec<Complex64> {
    let a_shape = convert_shape(a_shape);
    let a_strides = build_row_major_strides(&a_shape);
    let out_size = out_shape.iter().product();
    let out_shape = convert_shape(out_shape);
    let out_strides = build_row_major_strides(&out_shape);
    let mut out_data = Vec::with_capacity(out_size);

    let a_tensor = create_tensor(&a_shape, a_data, &a_strides);
    let mut c_tensor = create_out_tensor(&out_shape, out_data.as_ptr(), &out_strides);

    unsafe {
        tblis_sys::tblis_tensor_add(
            std::ptr::null(),
            std::ptr::null(),
            &raw const a_tensor.tensor,
            a_labels.as_ptr(),
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
    fn toy_contraction() {
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
            &[0, 1, 2],
            &[2, 3, 4],
            a_data,
            &[2],
            &[4],
            b_data,
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
    fn toy_contraction_transposed() {
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
            &[0, 1, 2],
            &[2, 3, 4],
            a_data,
            &[2],
            &[4],
            b_data,
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
    fn matrix_transpose() {
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ];
        let out = tensor_reorder(&[1, 0], &[3, 2], &[0, 1], &[2, 3], &data);

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
}
