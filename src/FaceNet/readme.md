# Tips


需要修改facenet 下 models 文件夹下 mtcnn.py 文件中 MTCNN 类的 forward 函数的返回值， 修改如下：


```python
if return_prob:
    return faces, batch_probs
else:
    return faces, batch_boxes
```