aws ec2 run-instances \
    --image-id ami-f0091d91 \
    --instance-type m4.large \
    --security-group jupyter-notebook \
    --key-name AWSFINAL
