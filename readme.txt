0准备    import
        常量定义
        生成数据集

1前向传播：定义输入、参数和输出
        x=
        y_=

        w1=
        w2=

        a=
        y=

2反向传播：定义损失函数、反向传播方法
        loss=
        train_step=

3生成会话，训练STEPS轮
with tf.session() as sess:
    init_op=tf.global_variables_initializer()
    sess_run(init_op)

    STEPS=3000
    for i in range（STEPS):
        start=
        end=
        sess.run(train_step,feed_dict:)

