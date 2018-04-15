package org.tfModelServing4s
package tf

import java.nio.{ByteBuffer, FloatBuffer}

import org.tensorflow.Tensor
import org.tfModelServing4s.dsl._


object implicits {

  object StringMapper {

    implicit val dim1Encoder = new TensorEncoder[String, Tensor, Array[Byte]] {

      def toTensor(data: Array[Byte], shape: List[Long]): Tensor[String] = {
        val t = Tensor.create(data)
        t.asInstanceOf[Tensor[String]]
      }
    }
  }

  object FloatMapper {
    implicit val dim1ArrayEncoder = new TensorEncoder[Float, Tensor, Array[Float]] {

      def toTensor(data: Array[Float], shape: List[Long]): Tensor[Float] = {
        val t = Tensor.create(shape.toArray, FloatBuffer.wrap(data))
        t.asInstanceOf[Tensor[Float]]
      }
    }

    implicit val dim1ArrayDecoder = new TensorDecoder[Float, Tensor, Array[Float]] {

      def fromTensor(tensor: Tensor[Float]): Array[Float] = {
        val shape = tensor.shape().toList.map(_.toInt)
        val array = Array.ofDim[Float](shape.head)
        tensor.copyTo(array)

        array
      }
    }

    implicit val dim2ArrayEncoder = new TensorEncoder[Array[Float], Tensor, Array[Array[Float]]] {

      def toTensor(data: Array[Array[Float]], shape: List[Long]): Tensor[Array[Float]] = {
        val t = Tensor.create(shape.toArray, FloatBuffer.wrap(data.flatten))
        t.asInstanceOf[Tensor[Array[Float]]]
      }
    }

    implicit val dim2ArrayDecoder = new TensorDecoder[Float, Tensor, Array[Array[Float]]] {

      def fromTensor(tensor: Tensor[Float]): Array[Array[Float]] = {
        val shape = tensor.shape().toList.map(_.toInt)
        val array = Array.ofDim[Float](shape.head, shape(1))
        tensor.copyTo(array)

        array
      }
    }
  }

  object ByteMapper {
    implicit val byte1DimArrayEncoder = new TensorEncoder[Byte, Tensor, Array[Byte]] {

      def toTensor(data: Array[Byte], shape: List[Long]): Tensor[Byte] = {
        val t = Tensor.create(shape.toArray, classOf[Byte])
        t.writeTo(ByteBuffer.wrap(data))
        t
      }
    }

    //  implicit val byte1DimArrayDecoder = new TensorDecoder[Byte, Tensor, Array[Byte]] {
    //
    //    def fromTensor(tensor: Tensor[Byte]) : Array[Byte] = {
    //      val shape = tensor.shape().toList.map(_.toInt)
    //      val array = Array.ofDim[Byte](shape.head)
    //      tensor.copyTo(array)
    //
    //      array
    //    }
    //  }

    implicit val byte2DimArrayDecoder = new TensorDecoder[Byte, Tensor, Array[Array[Float]]] {

      def fromTensor(tensor: Tensor[Byte]): Array[Array[Float]] = {
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
