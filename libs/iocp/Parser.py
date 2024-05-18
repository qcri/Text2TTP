#!/usr/bin/env python

###################################################################################################
#
# Copyright (c) 2015, Armin Buescher (armin.buescher@googlemail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###################################################################################################
#
# File:             iocp.py
# Description:      IOC Parser is a tool to extract indicators of compromise from security reports
#                   in PDF format.
# Usage:            iocp.py [-h] [-p INI] [-f FORMAT] PDF
# Author:           Armin Buescher (@armbues)
# Contributors:     Angelo Dell'Aera (@angelodellaera)
# Thanks to:        Jose Ramon Palanco
#                   Koen Van Impe (@cudeso)
#
###################################################################################################

import os
import glob
import re
from collections import defaultdict

try:
	import configparser as ConfigParser
except ImportError:
	import ConfigParser

# Import optional third-party libraries
IMPORTS = []

# Import project source files
from libs import iocp


class Parser(object):
	patterns = {}
	defang = {}

	def __init__(self, patterns_ini=None, input_format='str', dedup=False):
		basedir = iocp.get_basedir()

		if patterns_ini is None:
			patterns_ini = os.path.join(basedir, 'data/patterns.ini')
		self.load_patterns(patterns_ini)

		wldir = os.path.join(basedir, 'data/whitelists')
		self.whitelist = self.load_whitelists(wldir)

		self.dedup = dedup

		self.ext_filter = "*." + input_format
		parser_format = "parse_" + input_format
		try:
			self.parser_func = getattr(self, parser_format)
		except AttributeError:
			e = 'Selected parser format is not supported: %s' % (input_format)
			raise NotImplementedError(e)

	def load_patterns(self, fpath):
		config = ConfigParser.ConfigParser()
		with open(fpath) as f:
			config.read_file(f)

		for ind_type in config.sections():
			try:
				ind_pattern = config.get(ind_type, 'pattern')
			except:
				continue

			if ind_pattern:
				ind_regex = re.compile(ind_pattern)
				self.patterns[ind_type] = ind_regex

			try:
				ind_defang = config.get(ind_type, 'defang')
			except:
				continue

			if ind_defang:
				self.defang[ind_type] = True

	def load_whitelists(self, fpath):
		whitelist = {}

		searchdir = os.path.join(fpath, "whitelist_*.ini")
		fpaths = glob.glob(searchdir)
		for fpath in fpaths:
			t = os.path.splitext(os.path.split(fpath)[1])[0].split('_',1)[1]
			patterns = [line.strip() for line in open(fpath)]
			whitelist[t]  = [re.compile(p) for p in patterns]

		return whitelist

	def is_whitelisted(self, ind_match, ind_type):
		try:
			for w in self.whitelist[ind_type]:
				if w.findall(ind_match):
					return True
		except KeyError as e:
			pass
		return False

	def parse_page(self, data):
		iocs = defaultdict(list)
		for ind_type, ind_regex in self.patterns.items():
			matches = ind_regex.findall(data)

			for ind_match in matches:
				if isinstance(ind_match, tuple):
					ind_match = ind_match[0]

				if self.is_whitelisted(ind_match, ind_type):
					continue

				if ind_type in self.defang:
					ind_match = re.sub(r'\[\.\]', '.', ind_match)

				if self.dedup:
					if (ind_type, ind_match) in self.dedup_store:
						continue

					self.dedup_store.add((ind_type, ind_match))

				iocs[ind_type].append(ind_match)
		return iocs

	def parse_str(self, s):
		try:
			if self.dedup:
				self.dedup_store = set()

			return self.parse_page(s)
		except (KeyboardInterrupt, SystemExit):
			raise