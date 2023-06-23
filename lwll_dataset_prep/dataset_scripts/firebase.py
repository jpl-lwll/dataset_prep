# Copyright (c) 2023 California Institute of Technology (“Caltech”). U.S.
# Government sponsorship acknowledged.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Caltech nor its operating division, the Jet Propulsion
#   Laboratory, nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import firebase_admin
from firebase_admin import credentials, firestore
import os


class FB_Auth_Public(object):
    def __init__(self) -> None:
        # Use a service account
        cred = credentials.Certificate(
            f'{os.path.dirname(__file__)}/service_accounts/creds.json')
        _ = firebase_admin.initialize_app(cred)

        # First initialization can't have `name` argument, should really file bug on python SDK of firebase_admin repo
        self.db_firestore = firestore.client()
        self.db_realtime = ""
        pass

class FB_Auth_Private(object):
    def __init__(self) -> None:
        # Use a service account
        cred = credentials.Certificate(
            f'{os.path.dirname(__file__)}/service_accounts/creds2.json')
        app2 = firebase_admin.initialize_app(cred, name='fb_private')

        self.db_firestore = firestore.client(app2)
        self.db_realtime = ""
        pass


firebase = FB_Auth_Public()
fb_store_public = firebase.db_firestore
fb_realtime_public = firebase.db_realtime

firebase2 = FB_Auth_Private()
fb_store_private = firebase2.db_firestore
fb_realtime_private = firebase2.db_realtime
