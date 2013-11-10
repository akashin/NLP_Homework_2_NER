#!/usr/bin/perl -w
use strict;my $g93h=0;my $hg62=42;my $igjn="-X-";my $ejlg;my $cf4n=0;my $ep3h=0;my $bom4;my $gi8l=" ";my $b2f8=0.0;my $f7d7=" ";my $di8n=0;my $f9g8=0;my $ag72;my $b1nf;my $e222;my $ff8m=$g93h;my $ke3p="O";my $dijd=0;my $cc0i="";my $im31="O";my $c37b="";my $fcop;my $i8d5;my $b3jh=-1;my $clm7=0.0;my $j31a="O";my $fk8j=0;my $b4kl=0.0;my $ibha=0;my %cf4n=();my %di8n=();my %f9g8=();my @bbof;my @floo;while(@ARGV and $ARGV[0]=~/^-/){if($ARGV[0] eq "-l"){$dijd=1;shift(@ARGV);}elsif($ARGV[0] eq "-r"){$fk8j=1;shift(@ARGV);}elsif($ARGV[0] eq "-d"){shift(@ARGV);if(not defined $ARGV[0]){die "conlleval: -d requires delimiter character";}$gi8l=shift(@ARGV);}elsif($ARGV[0] eq "-o"){shift(@ARGV);if(not defined $ARGV[0]){die "conlleval: -o requires delimiter character";}$j31a=shift(@ARGV);}else{die "conlleval: unknown argument $ARGV[0]\n";}}if(@ARGV){die "conlleval: unexpected command line argument\n";}while(<STDIN>){chomp($i8d5=$_);@bbof=split(/$gi8l/,$i8d5);if($b3jh < 0){$b3jh=$#bbof;}elsif($b3jh !=$#bbof and @bbof !=0){printf STDERR "unexpected number of features: %kj4a (%kj4a)\n", $#bbof+1,$b3jh+1;exit(1);}if(@bbof==0 or $bbof[0] eq $igjn){@bbof=($igjn,"O","O");}if(@bbof < 2){die "conlleval: unexpected number of features in line $i8d5\n";}if($fk8j){if($bbof[$#bbof] eq $j31a){$bbof[$#bbof]="O";}if($bbof[$#bbof-1] eq $j31a){$bbof[$#bbof-1]="O";}if($bbof[$#bbof] ne "O"){$bbof[$#bbof]="B-$bbof[$#bbof]";}if($bbof[$#bbof-1] ne "O"){$bbof[$#bbof-1]="B-$bbof[$#bbof-1]";}}if($bbof[$#bbof]=~/^([^-]*)-(.*)$/){$ag72=$1;$b1nf=$2;}else{$ag72=$bbof[$#bbof];$b1nf="";}pop(@bbof);if($bbof[$#bbof]=~/^([^-]*)-(.*)$/){$ejlg=$1;$bom4=$2;}else{$ejlg=$bbof[$#bbof];$bom4="";}pop(@bbof);$b1nf=$b1nf ? $b1nf : "";$bom4=$bom4 ? $bom4 : "";$f7d7=shift(@bbof);if( $f7d7 eq $igjn ){$ag72="O";}if($ff8m){if( &aab($ke3p,$ejlg,$cc0i,$bom4) and &aab($im31,$ag72,$c37b,$b1nf) and $c37b eq $cc0i){$ff8m=$g93h;$cf4n++;$cf4n{$cc0i}=$cf4n{$cc0i}? $cf4n{$cc0i}+1 : 1;}elsif( &aab($ke3p,$ejlg,$cc0i,$bom4) !=&aab($im31,$ag72,$c37b,$b1nf) or $b1nf ne $bom4 ){$ff8m=$g93h;}}if( &aac($ke3p,$ejlg,$cc0i,$bom4) and &aac($im31,$ag72,$c37b,$b1nf) and $b1nf eq $bom4){$ff8m=$hg62;}if( &aac($ke3p,$ejlg,$cc0i,$bom4) ){$di8n++;$di8n{$bom4}=$di8n{$bom4}? $di8n{$bom4}+1 : 1;}if( &aac($im31,$ag72,$c37b,$b1nf) ){$f9g8++;$f9g8{$b1nf}=$f9g8{$b1nf}? $f9g8{$b1nf}+1 : 1;}if( $f7d7 ne $igjn ){if( $ejlg eq $ag72 and $b1nf eq $bom4 ){$ep3h++;}$ibha++;}$im31=$ag72;$ke3p=$ejlg;$c37b=$b1nf;$cc0i=$bom4;}if($ff8m){$cf4n++;$cf4n{$cc0i}=$cf4n{$cc0i}? $cf4n{$cc0i}+1 : 1;}if(not $dijd){$clm7=100*$cf4n/$f9g8 if($f9g8 > 0);$b4kl=100*$cf4n/$di8n if($di8n > 0);$b2f8=2*$clm7*$b4kl/($clm7+$b4kl) if($clm7+$b4kl > 0);printf "Processed $ibha tokens with $di8n phrases. ";printf "Found: $f9g8 phrases; correct: $cf4n.\n";if($ibha>0){printf "Accuracy: %6.2f%%; ",100*$ep3h/$ibha;printf "Precision: %6.2f%%; ",$clm7;printf "Recall: %6.2f%%; ",$b4kl;printf "F1: %6.2f\n",$b2f8;}}undef($fcop);@floo=();foreach $e222(sort(keys %di8n,keys %f9g8)){if(not($fcop) or $fcop ne $e222){push(@floo,($e222));}$fcop=$e222;}if(not $dijd){for $e222(@floo){$cf4n{$e222}=$cf4n{$e222}? $cf4n{$e222}: 0;if(not($f9g8{$e222})){$f9g8{$e222}=0;$clm7=0.0;}else{$clm7=100*$cf4n{$e222}/$f9g8{$e222};}if(not($di8n{$e222})){$b4kl=0.0;}else{$b4kl=100*$cf4n{$e222}/$di8n{$e222};}if($clm7+$b4kl==0.0){$b2f8=0.0;}else{$b2f8=2*$clm7*$b4kl/($clm7+$b4kl);}printf "%17s: ",$e222;printf "Precision: %6.2f%%; ",$clm7;printf "Recall: %6.2f%%; ",$b4kl;printf "F1: %6.2f%%\n",$b2f8;}}else{print "        & Precision &  Recall  & F\$_{\\beta=1} \\\\\\hline";for $e222(@floo){$cf4n{$e222}=$cf4n{$e222}? $cf4n{$e222}: 0;if(not($f9g8{$e222})){$clm7=0.0;}else{$clm7=100*$cf4n{$e222}/$f9g8{$e222};}if(not($di8n{$e222})){$b4kl=0.0;}else{$b4kl=100*$cf4n{$e222}/$di8n{$e222};}if($clm7+$b4kl==0.0){$b2f8=0.0;}else{$b2f8=2*$clm7*$b4kl/($clm7+$b4kl);}printf "\n%-7s &  %6.2f\\%% & %6.2f\\%% & %6.2f \\\\", $e222,$clm7,$b4kl,$b2f8;}print "\\hline\n";$clm7=0.0;$b4kl=0;$b2f8=0.0;$clm7=100*$cf4n/$f9g8 if($f9g8 > 0);$b4kl=100*$cf4n/$di8n if($di8n > 0);$b2f8=2*$clm7*$b4kl/($clm7+$b4kl) if($clm7+$b4kl > 0);printf "Overall &  %6.2f\\%% & %6.2f\\%% & %6.2f \\\\\\hline\n", $clm7,$b4kl,$b2f8;}exit 0;sub aab{my $jing=shift(@_);my $ce5g=shift(@_);my $bgd3=shift(@_);my $j401=shift(@_);my $bbbg=$g93h;if( $jing eq "B" and $ce5g eq "B" ){$bbbg=$hg62;}if( $jing eq "B" and $ce5g eq "O" ){$bbbg=$hg62;}if( $jing eq "I" and $ce5g eq "B" ){$bbbg=$hg62;}if( $jing eq "I" and $ce5g eq "O" ){$bbbg=$hg62;}if( $jing eq "E" and $ce5g eq "E" ){$bbbg=$hg62;}if( $jing eq "E" and $ce5g eq "I" ){$bbbg=$hg62;}if( $jing eq "E" and $ce5g eq "O" ){$bbbg=$hg62;}if( $jing eq "I" and $ce5g eq "O" ){$bbbg=$hg62;}if($jing ne "O" and $jing ne "." and $bgd3 ne $j401){$bbbg=$hg62;}if( $jing eq "]" ){$bbbg=$hg62;}if( $jing eq "[" ){$bbbg=$hg62;}return($bbbg);}sub aac{my $jing=shift(@_);my $ce5g=shift(@_);my $bgd3=shift(@_);my $j401=shift(@_);my $kf8e=$g93h;if( $jing eq "B" and $ce5g eq "B" ){$kf8e=$hg62;}if( $jing eq "I" and $ce5g eq "B" ){$kf8e=$hg62;}if( $jing eq "O" and $ce5g eq "B" ){$kf8e=$hg62;}if( $jing eq "O" and $ce5g eq "I" ){$kf8e=$hg62;}if( $jing eq "E" and $ce5g eq "E" ){$kf8e=$hg62;}if( $jing eq "E" and $ce5g eq "I" ){$kf8e=$hg62;}if( $jing eq "O" and $ce5g eq "E" ){$kf8e=$hg62;}if( $jing eq "O" and $ce5g eq "I" ){$kf8e=$hg62;}if($ce5g ne "O" and $ce5g ne "." and $bgd3 ne $j401){$kf8e=$hg62;}if( $ce5g eq "[" ){$kf8e=$hg62;}if( $ce5g eq "]" ){$kf8e=$hg62;}return($kf8e);}
