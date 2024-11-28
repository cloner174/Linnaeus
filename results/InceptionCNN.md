```
InceptionCNN(
  (layer1): InceptionBlock(
    (branch1): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
    (branch3): Sequential(
      (0): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (branch5): Sequential(
      (0): Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(16, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(3, 24, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (layer2): InceptionBlock(
    (branch1): Conv2d(88, 16, kernel_size=(1, 1), stride=(1, 1))
    (branch3): Sequential(
      (0): Conv2d(88, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (branch5): Sequential(
      (0): Conv2d(88, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(16, 24, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (branch_pool): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(88, 24, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (avg_pool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=88, out_features=5, bias=True)
)
```