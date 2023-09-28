import coverage
import sample_gpt

cov = coverage.Coverage()
cov.set_option("report:show_missing", True)
cov.start()
sample_gpt.test()
cov.stop()
cov.save()
cov.report()
