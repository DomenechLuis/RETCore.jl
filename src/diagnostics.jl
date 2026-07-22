#=
1.1 Métricas de bondad de ajuste:
    WAIC (Weighted Average Information Criterion):
        Es un criterio de información que penaliza la complejidad del modelo y se basa en ellikelihood. Útil para comparar modelos, especialmente cuando el sampler no ha convergido. 
    LOO (Leave-One-Out Cross-Validation):
        Une técnica de validación que mide el rendimiento del modelo dejando uno de los datos en cada iteración. Es robusto y puede usarse incluso si el sampler no converge.
    DIC (Deviance Information Criterion):
        Similar al WAIC, pero basado en la devianza. Útil para modelos bayesianos, pero menos general que el WAIC.
    PSIS-LOO:
        Una versión modificada del LOO que es más robusta a outliers y datosproblemáticos.



1.2 Pruebas de posterior predictivo:
    Generan predicciones basadas en los parámetros posteriores del modelo.

    Comparan estas predicciones con los datos observados para evaluar el ajuste del modelo.

    Ejemplo de uso:
        Graficar distribuciones KDE (Kernel Density Estimate) superpuestas de los datos observados y las predicciones posteriores.

1.3 Residuales y diagnóstico gráfico:
    Residuales:
        Calculan la diferencia entre los valores observados y los pronosticados.
        Se pueden plotear en diferentes escalas (log, lineal) para detectar patrones.
    QQ-plots:
        Comparan los residuales con distribuciones teóricas (normal, gamma, etc.).
        Útiles para evaluar si los errores siguen una determinada distribución.


diagnostic_plots = model_fit_diagnostic(fo, plots = [:waic, :ppc, :residuals])

=#

function prior_sensitivity(fo::FitObject)

    isnothing(fo.chains) && error("must be fitted before calling prior_sensitivity")

    priors = (
        wide = prior_wide,
        optimistic = prior_optimistic,
        pessimistic = prior_pessimistic,
        high_scatter = prior_high_scatter,
    )

    fos = Dict{Symbol,FitObject}()
    fos[:default] = FitObject(fo, copy_results = true)


    for (name, priorfun) in pairs(priors)
        foi = FitObject(fo)
        fit!(foi; prior = priorfun(fo.model))
        fos[name] = foi
    end



    return fos
end
