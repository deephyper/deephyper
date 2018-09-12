def foo(bar, baz, l):
  print('hello {0}'.format(bar))
  return 'foo' + bar

from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=1)
l = [i for i in range(10)]
async_result = pool.apply_async(foo, ('world', 'foo', l)) # tuple of args for foo

# do some other stuff in the main process

print(type(async_result))
return_val = async_result.get()  # get the return value from your function.
