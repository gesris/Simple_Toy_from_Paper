import CombineHarvester.CombineTools.ch as ch

cb = ch.CombineHarvester()
cb.AddObservations(["120"], ["ana"], ["era"], ["chan"], [(0, "cat")])
cb.AddProcesses(["120"], ["ana"], ["era"], ["chan"], ["sig"], [(0, "cat")], True)
cb.AddProcesses(["120"], ["ana"], ["era"], ["chan"], ["bkg"], [(0, "cat")], False)

cb.cp().process(["bkg"]).AddSyst(cb, "sys", "shape", ch.SystMap()(1.0))
cb.cp().ExtractShapes("shapes.root", "$PROCESS", "$PROCESS_$SYSTEMATIC")

"""
bbb = ch.BinByBinFactory()
bbb.SetVerbosity(1)
bbb.SetAddThreshold(0.0)
bbb.SetMergeThreshold(0.0)
bbb.SetFixNorm(True)
#bbb.MergeBinErrors(cb.cp())
bbb.AddBinByBin(cb.cp(), cb)
cb.SetGroup("bbb", [".*_bin_\\d+"])
cb.SetGroup("syst_plus_bbb", [".*"])
cb.SetGroup("syst", ["sys"])
"""

cb.PrintAll()

writer = ch.CardWriter("datacard.txt", "shapes_ch.root")
writer.SetVerbosity(1)
writer.CreateDirectories(False)
writer.WriteCards("", cb)