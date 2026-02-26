import torch
import torchreid
import os
def main():
    # 数据管理器
    datamanager = torchreid.data.ImageDataManager(
        root='E:/pylearning/reid-data',
        sources='market1501',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=64,
        transforms=['random_flip', 'random_crop'],
        workers=0  # Windows 必须设 0
    )

    # 模型，只用 softmax 分类
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='softmax',   # 仅分类
        pretrained=True
    )
    model = model.cuda()

    # 优化器
    optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.00035)
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='single_step', stepsize=20)

    # 训练引擎
    engine = torchreid.engine.ImageSoftmaxEngine(datamanager, model, optimizer, scheduler=scheduler)

    # 开始训练
    engine.run(
        max_epoch=1,
        save_dir='logs/resnet50',
        #save_interval=5
    )


if __name__ == "__main__":
    main()
