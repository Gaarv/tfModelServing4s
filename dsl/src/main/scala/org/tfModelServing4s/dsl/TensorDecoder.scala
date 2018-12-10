package org.tfModelServing4s.dsl

import scala.language.higherKinds


/**
  * Converts a tensor to its representation in the form of a data structure.
 *
  * @tparam Tensor Type of tensor to convert from.
  * @tparam TRepr Type of the representation to convert to e.g. Array, List etc.
  */
trait TensorDecoder[T, Tensor[_], TRepr] {

  def fromTensor(tensor: Tensor[T]): TRepr

}
