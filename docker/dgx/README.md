# Dockerfiles

Dockerfiles for the project:

  * `dgx/cudnn5` - cuddn5 from base dgx cuda image
  * `dgx/theano` - theano build from the `dgx/cudnn5` image
  * `ornl/tensorflow` - adding jupyter notebook to the base dgx tensorflow image
  * `ornl/theano` - adding keras to `dgx/theano` and other dependencies for the project.

```
make
```
