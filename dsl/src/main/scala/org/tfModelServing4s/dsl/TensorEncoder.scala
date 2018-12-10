package org.tfModelServing4s.dsl

import scala.language.higherKinds


/**
  * Builds a tensor from its representation in the form of a data structure.
  *
  * @tparam Tensor Type of the tensor to build.
  * @tparam TRepr Type of the representation to build tensor from e.g. Array, List etc.
  */
trait TensorEncoder[T, Tensor[_], TRepr] {

  def toTensor(data: TRepr, shape: List[Long]): Tensor[T]

}