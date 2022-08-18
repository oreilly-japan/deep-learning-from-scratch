# ゼロから作る Deep Learning


# Amazon SageMaker Studio Lab の使い方

[Amazon SageMaker Studio Lab](https://studiolab.sagemaker.aws/)は無料の機械学習環境です。事前の[メールアドレスによる登録](https://studiolab.sagemaker.aws/requestAccount)を行うと、JupyterLabの実行環境が利用可能です。

![SageMaker Studio のランディングページ](https://docs.aws.amazon.com/sagemaker/latest/dg/images/studio-lab-landing.png)

## Amazon SageMaker Studio Labを開始する
Studio Lab を利用開始するためには、アカウントのリクエストと作成が必要です。アカウントのリクエストはこのように行います。

1. [Studio Lab のランディングページ](https://studiolab.sagemaker.aws/) を開きます。
1. ["Request free account"](https://studiolab.sagemaker.aws/requestAccount) を選択します。
1. メールアドレスなど必要な情報を記入します。
1. "Submit request" ボタンを押します。
1. メールアドレス確認のためのEメールを受け取ったら、案内に従って設定を完了してください。

以下で Studio Lab のアカウント作成を行う前に、アカウントリクエストが承認される必要があります。リクエストは 5 営業日以内に審査されます。詳細は[ドキュメント (英語)](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-onboard.html) をご覧ください。

Studio Lab アカウントの作成は、以下の手順で行います。
1. リクエスト承認メール内の "Create account" を押しページを開きます。
1. Eメールアドレス、パスワード、ユーザー名を入力します。
1. "Create account" を選択します。

Studio Lab へのサインインは、
1. [Studio Lab のランディングページ](https://studiolab.sagemaker.aws/) を開き、
1. 右上の "Sign in" ボタンを押します。
1. Eメールアドレス、パスワード、ユーザー名を入力します。
1. "Sign in" を選択しプロジェクトのページを開きます。

## CPU/GPUを使用する
Studio Lab では12時間のcompute timeのあいだ CPU インスタンス、4時間の compute time のあいだ GPU インスタンスを連続して利用することができます。なお、15 GB のストレージが割り当てられるので、ダウンロードしたデータや実行したコード、保存したファイルなどは、後のサインイン時に引き続き利用することができます。コンピュートインスタンスの詳細は[ドキュメント (英語)](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-overview.html#studio-lab-overview-project-compute) をご覧下さい。
1. Studio Lab にサインインしたら、このようなプロジェクトページが表示されます。
1. "My Project" 以下の "Select compute type" から `CPU`か`GPU` を選択します。
1. "Start runtime" を押します。
1. ランタイムが開始したら "Open project" をクリックし JupyterLab 環境を開きます。

![Studio Lab Project](https://docs.aws.amazon.com/sagemaker/latest/dg/images/studio-lab-overview.png)

## コードを実行する
Studio Lab では JupyterLab のインターフェイスを拡張した UI が提供されています。JupyterLab の UI になじみのない方は [The JupyterLab Interface](https://jupyterlab.readthedocs.io/en/latest/user/interface.html) のページをご覧ください。

![SageMaker Studio UI](https://docs.aws.amazon.com/sagemaker/latest/dg/images/studio-lab-ui.png)

## 外部ストレージ (Amazon S3) や Amazon SageMaker Studio の利用
Studio Lab の project に割り当てられた 15 GB のストレージを超えて利用したい場合は、[Amazon S3 に接続](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-external.html#studio-lab-use-external-s3)するか、[Amazon SageMaker Studio への移行](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-migrate.html) を検討してください。
