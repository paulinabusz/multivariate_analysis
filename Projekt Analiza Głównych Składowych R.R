dane<-Dane_transport[2:8]
dane

#dane podane sa w róznych jednostkach w zwiazku z czym poddaje je standaryzacji

dane_standaryzowane<-scale(dane)
dane_standaryzowane
#sprawdzam czy zmienne sa ze soba skorelowane

cor(dane_standaryzowane)

library(psych)
library(stats)

#Obliczenie statystyk opisowych

opis<-describe(dane_standaryzowane)
opis

#Obliczenie macierzy kowariancji oraz jej wartosci wlasnych

macierz_kowariancji<-cov(dane_standaryzowane)
eigen(macierz_kowariancji)


#obliczenie glównych skladowych

fit <- princomp(dane_standaryzowane, cor=FALSE)
fit
summary(fit) #statystyki wyjasnionej wariancji
fit$scores

#Obliczenie ladunków

ladunki<-loadings(fit) 
print(ladunki, cutoff = 0.001)
plot(fit,type="lines",lwd=3,col="pink",cex.lab=1.4,main="wykres osypiska") # wykres osypiska

biplot(fit)

#regresja dla skladowych glównych

Dane_2<-Railway_transport_passengers
y_niestandaryzowany<-Railway_transport_passengers[,2]
y<-scale(Dane_2[,2])
skladowe_glówne<-fit$scores
x1<-skladowe_glówne[,1]
x2<-skladowe_glówne[,2]
regresja_1<-lm(y~ -1 + x1+x2)
regresja_1$coefficients
summary(regresja_1)
regresja_1$residuals


#funkcja pcr

library(pls) 

pcr_model<-pcr(y ~ -1 + x1 + x2)
pcr_model$coefficients
summary(pcr_model)
pcr_model$coefficients
pcr_model$residuals

#badanie normalnoœci za pomoc¹ testu Shapiro-Wilka
pcr_residuals<-pcr_model$residuals
shapiro.test(pcr_model$residuals)

#badanie homoskedastycznoœci wariancji za pomoc¹ testu Breuscha-Pagana

library(lmtest)

bptest(pcr_model)



#regresja dla danych stadnaryzowanych

z1<-dane[,1]
z2<-dane[,2]
z3<-dane[,3]
z4<-dane[,4]
z5<-dane[,5]
z6<-dane[,6]
z7<-dane[,7]

regresja_2<-lm(y_niestandaryzowany~ -1 +z1+z2+z3+z4+z5+z6+z7)
summary(regresja_2)

regresja_2$coefficients
regresja_3_residuals<-regresja_3$residuals
shapiro.test(regresja_3_residuals)

library(lmtest)
bptest(regresja_2)

#Kryterium informacyjne

AIC(regresja_2)
AIC(regresja_1)

######## Analiza g³ównych sk³adowych dla danych scentrowanych ########

#scentrowanie danych
dane_scentrowane<-scale(dane,center=TRUE, scale=FALSE)

#Sprawdzenie korelacji
cor(dane_scentrowane)

#macierz kowariancji oraz wartoœci w³asne
macierz_kowariancji_2<-cov(dane_scentrowane)
eigen(macierz_kowariancji_2)


#obliczenie glównych skladowych

fit_2 <- princomp(dane_scentrowane, cor=FALSE)
fit_2
summary(fit_2) #statystyki wyjasnionej wariancji
fit$scores

#Obliczenie ladunków

ladunki<-loadings(fit_2) 
print(ladunki, cutoff = 0.000001)
plot(fit,type="lines",lwd=3,col="pink",cex.lab=1.4,main="wykres osypiska")