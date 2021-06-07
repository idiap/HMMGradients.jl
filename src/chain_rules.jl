# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

function ChainRulesCore.rrule(::typeof(nlogML), Nt, a, A, y)
  alpha, c = forward(Nt,a,A,y)
  function nlogML_pullback(delta)
    grada, gradA, grady = get_grad(Nt,a,A,y,alpha,c)
    return NoTangent(), NoTangent(), grada, gradA, grady
  end
  return nlogML(Nt,c), nlogML_pullback
end

function ChainRulesCore.rrule(::typeof(nlogMLlog), Nt, a, A, y)
  alpha, logML = logforward(Nt,a,A,y)
  function nlogMLlog_pullback(delta)
    grada, gradA, grady = get_loggrad(Nt,a,A,y,alpha,logML)
    return NoTangent(), NoTangent(), grada, gradA, grady
  end
  return -sum(logML), nlogMLlog_pullback
end
