package Xreducer_fuzzy;

public class SimilarityStyle {
	public double Vmax = 0;
	public double Vmin = 0;
	public double Vstdvar = 0;
	public void SimilaritySetting(double Vmax,double Vmin,double Vstdvar){
		this.Vmax = Vmax;
		this.Vmin = Vmin;
		this.Vstdvar = Vstdvar;
	}
	public double getSimilarityValue(double x, double y){
		return 0;
	}
	public String getInformation(){
		return null;
	}
	public String getInfor(){
		return null;
	}
}
