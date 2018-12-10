package org.tfModelServing4s
package tf

import java.nio.{ByteBuffer, FloatBuffer}

import org.tensorflow.Tensor
import org.tfModelServing4s.dsl._


object implicits {

  object StringMapper {

    private type T = String

    implicit val dim1Encoder = new TensorEncoder[T, Tensor, Array[Byte]] {

      def toTensor(data: Array[Byte], shape: List[Long]): Tensor[T] =
        Tensor.create(data, classOf[T])
    }
  }

  object FloatMapper {

    private type T = Float

    implicit val dim1ArrayEncoder = new TensorEncoder[T, Tensor, Array[T]] {

      def toTensor(data: Array[T], shape: List[Long]): Tensor[T] = {
        val t = Tensor.create(shape.toArray, classOf[T])
        t.writeTo(FloatBuffer.wrap(data))
        t
      }
    }

    implicit val dim1ArrayDecoder = new TensorDecoder[T, Tensor, Array[T]] {

      def fromTensor(tensor: Tensor[T]): Array[T] = {
        val shape = tensor.shape().toList.map(_.toInt)
        val array = Array.ofDim[T](shape.head)
        tensor.copyTo(array)

        array
      }
    }

    implicit val dim2ArrayEncoder = new TensorEncoder[Array[T], Tensor, Array[Array[T]]] {

      def toTensor(data: Array[Array[T]], shape: List[Long]): Tensor[Array[T]] = {
        val t = Tensor.create(shape.toArray, classOf[Array[T]])
        t.writeTo(FloatBuffer.wrap(data.flatten))
        t
      }
    }

    implicit val dim2ArrayDecoder = new TensorDecoder[T, Tensor, Array[Array[T]]] {

      def fromTensor(tensor: Tensor[T]): Array[Array[T]] = {
        val shape = tensor.shape().toList.map(_.toInt)
        val array = Array.ofDim[T](shape.head, shape(1))
        tensor.copyTo(array)

        array
      }
    }
  }

  object ByteMapper {

    private type T = Byte

    implicit val dim1ArrayEncoder = new TensorEncoder[T, Tensor, Array[T]] {

      def toTensor(data: Array[T], shape: List[Long]): Tensor[T] = {
        val t = Tensor.create(shape.toArray, classOf[T])
        t.writeTo(ByteBuffer.wrap(data))
        t
      }
    }

    //  implicit val dim1ArrayDecoder = new TensorDecoder[T, Tensor, Array[T]] {
    //
    //    def fromTensor(tensor: Tensor[T]) : Array[T] = {
    //      val shape = tensor.shape().toList.map(_.toInt)
    //      val array = Array.ofDim[T](shape.head)
    //      tensor.copyTo(array)
    //
    //      array
    //    }
    //  }

    implicit val dim2ArrayDecoder = new TensorDecoder[T, Tensor, Array[Array[Float]]] {

      def fromTensor(tensor: Tensor[T]): Array[Array[Float]] = {
        val shape = tensor.shape().toList.map(_.toInt)
        val array = Array.ofDim[Float](shape.head, shape(1))
        tensor.copyTo(array)

        array
      }
    }
  }

  implicit def closeableTensor[T] = new Closeable[Tensor[T]] {

    def close(resource: Tensor[T]): Unit = {

      println("releasing TF tensor")
      resource.close()
    }

  }

  implicit val closeableModel = new Closeable[TFModel] {

    def close(resource: TFModel): Unit = {

      println("closing TF model")
      resource.bundle.close()
    }
  }
}
