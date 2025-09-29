package ml.combust.mleap.tensorflow.converter

import ml.combust.mleap.core.types.{BasicType, TensorType}

import ml.combust.mleap.tensor.{ByteString, DenseTensor}
import org.tensorflow
import org.tensorflow.ndarray.{NdArray, NdArraySequence}
import org.tensorflow.types._


import scala.collection.mutable.ArrayBuffer
import java.util.function.BiConsumer


/**
  * Created by hollinwilkins on 1/12/17.
  */
object TensorflowConverter {
    def convert(tensor: tensorflow.Tensor, tensorType: TensorType): DenseTensor[_] = {
      val dimensions = getShape(tensor)
      val size = getSize(tensor)

      tensor match {
        case u8: TUint8 =>
          val buffer = u8.asRawTensor().data()
          val array = new Array[Byte](buffer.size().toInt)
          buffer.read(array)
          DenseTensor(array, dimensions)
        case i32: TInt32 =>
          val data = i32.asRawTensor().data().asInts()
          val array = new Array[Int](size)
          data.read(array)
          DenseTensor(array, dimensions)
        case i64: TInt64=>
          val data = i64.asRawTensor().data().asLongs()
          val array = new Array[Long](size)
          data.read(array)
          DenseTensor(array, dimensions)
        case f32: TFloat32 =>
          val data = f32.asRawTensor().data().asFloats()
          val array = new Array[Float](size)
          data.read(array)
          DenseTensor(array, dimensions)
        case f64: TFloat64 =>
          val data = f64.asRawTensor().data().asDoubles()
          val array = new Array[Double](size)
          data.read(array)
          DenseTensor(array, dimensions)
        case str: TString =>
          tensorType.base match {
            case BasicType.String =>
              val array = ArrayBuffer[String]()
              if (dimensions.isEmpty) {
                array += str.getObject()
              } else {
                str.scalars.asInstanceOf[NdArraySequence[NdArray[String]]].forEachIndexed(new BiConsumer[Array[Long], NdArray[String]] {
                  override def accept(i: Array[Long], e: NdArray[String]): Unit = {
                    array += e.getObject()
                  }
                })
              }
              DenseTensor(array.toArray, dimensions)
            case BasicType.ByteString =>
              val array = ArrayBuffer[ByteString]()
              if (dimensions.isEmpty) {
                array += ByteString(str.asBytes().getObject())
              } else {
                str.asBytes().scalars.asInstanceOf[NdArraySequence[NdArray[Array[Byte]]]].forEachIndexed(new BiConsumer[Array[Long], NdArray[Array[Byte]]] {
                  override def accept(i: Array[Long], e: NdArray[Array[Byte]]): Unit = {
                    array += ByteString(e.getObject())
                  }
                })
              }
              DenseTensor(array.toArray, dimensions)
            case _ =>
              throw new RuntimeException(s"unsupported ml TensorType ${tensorType} when Tensorflow tensor is String")
          }
        case _ =>
          throw new RuntimeException(s"unsupported tensorflow type: ${tensor.dataType()}")
      }
    }

  def getSize(tensor: tensorflow.Tensor): Int = {
    tensor.shape.size.asInstanceOf[Int]
  }

  def getShape(tensor: tensorflow.Tensor): Seq[Int] = {
    tensor.shape.asArray.toSeq.map(_.toInt)
  }
}
