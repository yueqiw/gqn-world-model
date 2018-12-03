REMOTE="gilbert@35.202.179.29"

rsync -aP ${REMOTE}:gqn-pytorch/output local_path

REMOTE='yueqi@100.33.245.120'
rsync -aP -e "ssh -p 4282" gqn-pytorch/output ${REMOTE}:Dropbox/lib/gqn-pytorch/output_remote

