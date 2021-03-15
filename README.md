# С использованием примера [2] обучить нейронную сеть EfficientNet-B0 [3,4,5] (случайное начальное приближение) для решения задачи классификации изображений Oregon WildLife [6]
-Графики обучения:
 - Валидация - синий цвет
 - Тренировка - оранжевый цвет
   
   График метрики качества
  ![SVG example](./epoch_categorical_accuracy.svg)
  
  График функции потерь
  ![SVG example](./epoch_loss.svg)


# С использованием [2] и техники обучения Transfer Learning [7] обучить нейронную сеть EfficientNet-B0 (предобученную на базе изображений imagenet) для решения задачи классификации изображений Oregon WildLife
-Архитектура:

```
   inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = EfficientNetB0(include_top=False, input_tensor=inputs, pooling='avg', weights='imagenet')(inputs)
  
  x.trainable = False
  
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)

  return tf.keras.Model(inputs=inputs, outputs=outputs)
```

-Графики обучения:
 - Валидация - синий цвет
 - Тренировка - оранжевый цвет
   
   График метрики качества
  ![SVG example](./epoch_categorical_accuracy_2.svg)
  
  График функции потерь
  ![SVG example](./epoch_loss_2.svg)

 # Анализ результатов
 
